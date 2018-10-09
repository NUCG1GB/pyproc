# Copyright 2014-2017 United Kingdom Atomic Energy Authority
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.


# External imports
import os
from scipy.constants import electron_mass, atomic_mass

import matplotlib.pyplot as plt
import numpy as np
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung
from cherab.core.model.lineshape import StarkBroadenedLine
from cherab.core.math import Constant3D, ConstantVector3D

# Cherab and raysect imports
from cherab.core import Species, Maxwellian, Plasma, Line, elements
from cherab.openadas import OpenADAS
from cherab.openadas.models.continuo import Continuo
from demos.uniform_volume import UniformVolume

# Core and external imports
from raysect.optical import World, translate, rotate, Vector3D, Point3D, Ray
from raysect.primitive import Sphere, Cylinder
from raysect.optical.observer import PinholeCamera
from raysect.core.workflow import SerialEngine
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

# tunables
ion_density = 1e21
sigma = 0.25

# setup scenegraph
world = World()

# create atomic data source
adas = OpenADAS(permit_extrapolation=True)

# PLASMA ----------------------------------------------------------------------
plasma = Plasma(parent=world)
plasma.atomic_data = adas
plasma.geometry = Sphere(0.5)
plasma.geometry_transform = None
plasma.integrator = NumericalIntegrator(step=sigma / 5.0)

# define basic distributions
d_density = UniformVolume(ion_density)
e_density = UniformVolume(ion_density)
temperature = UniformVolume(1.)
bulk_velocity = ConstantVector3D(Vector3D(0, 0, 0))

d_distribution = Maxwellian(d_density, temperature, bulk_velocity, elements.deuterium.atomic_weight * atomic_mass)
e_distribution = Maxwellian(e_density, temperature, bulk_velocity, electron_mass)

d0_species = Species(elements.deuterium, 0, d_distribution)
d1_species = Species(elements.deuterium, 1, d_distribution)

# define species
plasma.b_field = ConstantVector3D(Vector3D(1.0, 1.0, 1.0))
plasma.electron_distribution = e_distribution
plasma.composition = [d0_species, d1_species]

# Setup elements.deuterium lines
d_delta = Line(elements.deuterium, 0, (6, 2))

plasma.models = [
    Continuo(),
    ExcitationLine(d_delta, lineshape=StarkBroadenedLine),
    RecombinationLine(d_delta, lineshape=StarkBroadenedLine)
]

# alternate geometry
# plasma.geometry = Cylinder(sigma * 2.0, sigma * 10.0)
# plasma.geometry_transform = translate(0, -sigma * 5.0, 0) * rotate(0, 90, 0)

plt.ion()

r = Ray(origin=Point3D(0, 0, -1), direction=Vector3D(0, 0, 1), min_wavelength=300, max_wavelength=415, bins=10000)
s = r.trace(world)
plt.semilogy(s.wavelengths, s.samples)

# plt.show()

# camera = PinholeCamera((128, 128), parent=world, transform=translate(0, 0, -2.))
# # camera.render_engine = SerialEngine()
# camera.spectral_rays = 1
# camera.spectral_bins = 15
# camera.pixel_samples = 50
#
plt.ion()
# camera.observe()
#
plt.ioff()
plt.show()
