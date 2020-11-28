"""
Top-down car dynamics simulation.

Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
This simulation is a bit more detailed, with wheels rotation.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""

import numpy as np
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)

SIZE = 0.02
ENGINE_POWER = 100000000*SIZE*SIZE
WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
FRICTION_LIMIT = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_WEAR_COEFF = 0.001 # can be in range of (0-10] (can be higher but that gives wonky results)
WHEEL_K = WHEEL_WEAR_COEFF*5000 - 1 # number should 5*track length
WHEEL_BASE_FRICTION = 1
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [
    (-55, +80), (+55, +80),
    (-55, -82), (+55, -82)
    ]
HULL_POLY1 = [
    (-60, +130), (+60, +130),
    (+60, +110), (-60, +110)
    ]
HULL_POLY2 = [
    (-15, +120), (+15, +120),
    (+20, +20), (-20, 20)
    ]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20)
    ]
HULL_POLY4 = [
    (-50, -120), (+50, -120),
    (+50, -90),  (-50, -90)
    ]
WHEEL_COLOR = (0.0,  0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)
MUD_COLOR = (0.4, 0.4, 0.0)
MAXSPEED = 100
MAXACEL = 40
MAXDECEL = 50


class Car:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY1]), density=1.0),
                fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY2]), density=1.0),
                fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY3]), density=1.0),
                fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY4]), density=1.0)
                ]
            )
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)
            ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x+wx*SIZE, init_y+wy*SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x*front_k*SIZE,y*front_k*SIZE) for x, y in WHEEL_POLY]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
                    )
            w.wheel_rad = front_k*WHEEL_R*SIZE
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*SIZE, wy*SIZE),
                localAnchorB=(0,0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180*900*SIZE*SIZE,
                motorSpeed=0,
                lowerAngle=-0.4, # wheel rotation angles?
                upperAngle=+0.4,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

        self.totalDistance = 0
        self.tireWear = 0

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1: diff = 0.1  # gradually increase, but stop immediately (replace with acceleration curve)
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):

        def acel(gas, rad_of_wheel, v, dt):
            current_vel = math.sqrt(v.x**2 + v.y**2)
            scaledVelocity = current_vel*10/MAXSPEED
            k = (10**(-5))
            acel_coef = 1/(k*scaledVelocity**7+1)
            currentAcc = gas*MAXACEL*acel_coef
            current_vel = current_vel + currentAcc*dt 
            omega = current_vel/rad_of_wheel
            return (omega)
        
        def decel(brake, rad_of_wheel, v, dt):
            current_vel = math.sqrt(v.x**2 + v.y**2)
            scaledVelocity = current_vel*27/MAXSPEED
            a = -0.0000336765 # or -33.6765*10^-6
            b = 0.00183713
            c = -0.0365703
            d = 0.313523
            decel_coef = a*scaledVelocity**4 + b*scaledVelocity**3 + c*scaledVelocity**2 + d*scaledVelocity
            currentDecel = brake*MAXDECEL*decel_coef
            current_vel = current_vel - currentDecel*dt
            omega = current_vel/rad_of_wheel
            return (omega)
        tire = 1 #(Harder)--> C1,C2,C3,C4,C5 <--(Softest)
        if tire == 1:
            WHEEL_WEAR_COEFF = 0.0004 # Depending the speed should last about 2 laps
            WHEEL_K= WHEEL_WEAR_COEFF * 1 # Starts degrading from the get go
            FRICTION_LIMIT = 1000000*SIZE*SIZE
        elif tire == 2:
            WHEEL_WEAR_COEFF = 0.0006 # Lasts ~1.75 Laps
            WHEEL_K = WHEEL_WEAR_COEFF * 1
            FRICTION_LIMIT = 1200000 * SIZE * SIZE
        elif tire == 3:
            WHEEL_WEAR_COEFF = 0.0007 # Lasts ~1.5 Laps
            WHEEL_K = WHEEL_WEAR_COEFF * 1
            FRICTION_LIMIT = 1400000 * SIZE * SIZE
        elif tire == 4:
            WHEEL_WEAR_COEFF = 0.0009 # Lasts 1 Laps
            WHEEL_K = WHEEL_WEAR_COEFF * 1  # Improves the amount of friction that is remaining
            FRICTION_LIMIT = 1600000 * SIZE * SIZE
        else:
            WHEEL_WEAR_COEFF = 0.0010 # Barely completes 1 lap
            WHEEL_K = WHEEL_WEAR_COEFF * 1
            FRICTION_LIMIT = 1800000 * SIZE * SIZE
        topthing = WHEEL_WEAR_COEFF * self.totalDistance - WHEEL_K
        botthing = 1 + abs(WHEEL_WEAR_COEFF * self.totalDistance - (WHEEL_K + 1))
        self.tireWear = 1 - (.5 * (topthing/botthing + 1))

        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir*min(50.0*val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT*0.6*self.tireWear * WHEEL_BASE_FRICTION  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, FRICTION_LIMIT*tile.road_friction) * self.tireWear * WHEEL_BASE_FRICTION
                grass = False
            
            friction_limit = max(friction_limit, 20)
            # print("Friction Limit: {}".format(friction_limit))

            # Force
            forw = w.GetWorldVector( (0,1) )
            side = w.GetWorldVector( (1,0) )
            v = w.linearVelocity
            vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed
            vs = side[0]*v[0] + side[1]*v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            # w.omega += dt*ENGINE_POWER*w.gas/WHEEL_MOMENT_OF_INERTIA/(abs(w.omega)+5.0)
            w.omega = acel(w.gas, w.wheel_rad, v, dt)
            self.fuel_spent += dt*ENGINE_POWER*w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                w.omega = decel(w.brake, w.wheel_rad, v, dt)
            w.phase += w.omega*dt

            vr = w.omega*w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr        # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000*SIZE*SIZE
            p_force *= 205000*SIZE*SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            # print(force)
            if abs(force) > 2.0*friction_limit:
                if w.skid_particle and w.skid_particle.grass == grass and len(w.skid_particle.poly) < 30:
                    w.skid_particle.poly.append( (w.position[0], w.position[1]) )
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle( w.skid_start, w.position, grass )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt*f_force*w.wheel_rad/WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter( (
                p_force*side[0] + f_force*forw[0],
                p_force*side[1] + f_force*forw[1]), True )

            magnitude = math.sqrt(w.linearVelocity.x**2 + w.linearVelocity.y**2)
            self.totalDistance += dt * magnitude / 4

        # print("Tire Wear: {}".format(self.tireWear))
        # print("Total Distance: {}".format(self.totalDistance))

    def draw(self, viewer, draw_particles=True):
        if draw_particles:
            for p in self.particles:
                viewer.draw_polyline(p.poly, color=p.color, linewidth=5)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)
                if "phase" not in obj.__dict__: continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0: continue
                if s1 > 0: c1 = np.sign(c1)
                if s2 > 0: c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W*SIZE, +WHEEL_R*c1*SIZE), (+WHEEL_W*SIZE, +WHEEL_R*c1*SIZE),
                    (+WHEEL_W*SIZE, +WHEEL_R*c2*SIZE), (-WHEEL_W*SIZE, +WHEEL_R*c2*SIZE)
                    ]
                viewer.draw_polygon([trans*v for v in white_poly], color=WHEEL_WHITE)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []

