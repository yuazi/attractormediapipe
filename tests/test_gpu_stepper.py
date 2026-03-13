from __future__ import annotations

import unittest

import numpy as np

from attractors.manager import active_attractor_names, create_active_attractor
from renderer.gpu_stepper import TransformFeedbackTrailStepper, generate_cpu_samples


class GpuStepperTests(unittest.TestCase):
    def test_cpu_helper_matches_attractor_fill_samples_for_all_active_attractors(self) -> None:
        for name in active_attractor_names():
            attractor = create_active_attractor(name)
            expected = attractor.fill_samples(0.005, 16).copy()

            attractor = create_active_attractor(name)
            actual = generate_cpu_samples(
                name,
                tuple(float(value) for value in attractor.state),
                0.005,
                attractor.parameter_dict(),
                16,
            )

            self.assertTrue(np.allclose(actual, expected, atol=1e-5), name)

    def test_transform_feedback_matches_cpu_samples_when_context_is_available(self) -> None:
        try:
            import moderngl

            ctx = moderngl.create_standalone_context(require=330)
        except Exception as exc:  # pragma: no cover - depends on local OpenGL availability
            self.skipTest(f"Standalone OpenGL context unavailable: {exc}")

        stepper = TransformFeedbackTrailStepper.create_if_supported(ctx)
        if stepper is None:  # pragma: no cover - guarded by context creation
            self.skipTest("Transform feedback stepper is unavailable for this OpenGL context")

        try:
            attractor = create_active_attractor("Lorenz")
            expected = generate_cpu_samples(
                attractor.name,
                tuple(float(value) for value in attractor.state),
                0.005,
                attractor.parameter_dict(),
                12,
            )
            actual, final_state = stepper.generate(
                attractor_name=attractor.name,
                state=tuple(float(value) for value in attractor.state),
                dt=0.005,
                parameters=attractor.parameter_dict(),
                steps=12,
            )

            self.assertTrue(np.allclose(actual, expected, atol=1e-5))
            self.assertTrue(np.allclose(actual[-1], np.asarray(final_state, dtype=np.float32), atol=1e-5))
        finally:
            stepper.release()
            ctx.release()


if __name__ == "__main__":
    unittest.main()
