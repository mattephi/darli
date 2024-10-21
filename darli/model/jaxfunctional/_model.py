from typing import Dict, List

import casadi as cs
from jaxadi import convert

from darli.backend import BackendBase, CasadiBackend, CentroidalDynamics
from darli.utils.arrays import ArrayLike

from .._base import CoM, Energy
from .._body import Body
from .._model import Model, ModelBase
from ..functional import Functional
from ..functional._body import FunctionalBody

# TODO: fix returns
# TODO: check all methods

class JaxFunctional(ModelBase):
    def __init__(self, model: Functional):
        self.functional = model

    @property
    def expression_model(self):
        return self.functional.__model

    @property
    def q(self) -> ArrayLike:
        return self.functional.__model.q

    @property
    def v(self) -> ArrayLike:
        return self.functional.__model.v

    @property
    def dv(self) -> ArrayLike:
        return self.functional.__model.dv

    @property
    def qfrc_u(self) -> ArrayLike:
        return self.functional.__model.qfrc_u

    @property
    def backend(self) -> BackendBase:
        return self.functional.__model.backend

    @property
    def nq(self) -> int:
        return self.functional.__model.backend.nq

    @property
    def nv(self) -> int:
        return self.functional.__model.backend.nv

    @property
    def nu(self) -> int:
        return self.functional.__model.nu

    @property
    def nbodies(self) -> int:
        return self.functional.__model.backend.nbodies

    @property
    def q_min(self) -> ArrayLike:
        return self.functional.__model.backend.q_min

    @property
    def q_max(self) -> ArrayLike:
        return self.functional.__model.backend.q_max

    @property
    def joint_names(self) -> List[str]:
        return self.functional.__model.backend.joint_names

    @property
    def bodies(self) -> Dict[str, FunctionalBody]:
        # TODO: probably we should map each element to FunctionalBody too
        return self.functional.__model.bodies

    def add_body(self, bodies_names: List[str] | Dict[str, str]):
        return self.functional.__model.add_body(bodies_names, Body)

    def body(self, name: str) -> FunctionalBody:
        return FunctionalBody.from_body(self.functional.__model.body(name))

    # @property
    # def state_space(self):
    #     return FunctionalStateSpace.from_space(self.__model.state_space)

    @property
    def selector(self):
        return self.functional.__model.selector

    def joint_id(self, name: str) -> int:
        return self.functional.__model.joint_id(name)

    @property
    def contact_forces(self) -> List[ArrayLike]:
        return self.functional.__model.contact_forces

    @property
    def contact_names(self) -> List[str]:
        return self.functional.__model.contact_names

    def update_selector(
        self,
        matrix: ArrayLike | None = None,
        passive_joints: List[str | int] | None = None,
    ):
        self.functional.__model.update_selector(matrix, passive_joints)

    @property
    def gravity(self) -> cs.Function:
        return convert(self.functional.gravity)

    @property
    def com(self) -> CoM:
        return convert(self.functional.com)

    @property
    def energy(self) -> Energy:
        return Energy(
            kinetic=convert(self.functional.energy.kinetic),
            potential=convert(self.functional.energy.potential)
        )

    @property
    def inertia(self) -> ArrayLike:
        return convert(self.functional.inertia)

    @property
    def coriolis(self) -> ArrayLike:
        return convert(self.functional.coriolis)

    @property
    def bias_force(self) -> ArrayLike:
        return convert(self.functional.bias_force)

    @property
    def momentum(self) -> ArrayLike:
        return convert(self.functional.momentum)

    @property
    def lagrangian(self) -> ArrayLike:
        return convert(self.functional.lagrangian)

    @property
    def contact_qforce(self) -> ArrayLike:
        return convert(self.functional.contact_qforce)

    @property
    def coriolis_matrix(self) -> ArrayLike:
        return convert(self.functional.coriolis_matrix)

    @property
    def forward_dynamics(self) -> ArrayLike:
        return convert(self.functional.forward_dynamics)

    @property
    def inverse_dynamics(self) -> ArrayLike:
        return convert(self.functional.inverse_dynamics)

    @property
    def centroidal_dynamics(self) -> CentroidalDynamics:
        return convert(self.functional.centroidal_dynamics)

    def update(
        self,
        q: ArrayLike,
        v: ArrayLike,
        dv: ArrayLike | None = None,
        u: ArrayLike | None = None,
    ) -> ArrayLike:
        # dummy implementation to satisfy base class
        return
