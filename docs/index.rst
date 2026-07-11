MADMAX Closed-Loop Calibration
==============================

Implementation of the **seven-step closed-loop calibration algorithm** for
the MADMAX dielectric-haloscope detector, as specified in the project's
design documents (included verbatim under :doc:`design/index`).

The system starts from an already-optimized nominal disk configuration,
treats the existing gradient-method boost-factor determination as the
high-fidelity experimental objective, and iteratively finds the best
calibrated real-detector configuration using *physics-informed, safe,
budget-aware Bayesian optimization* together with a *joint
detector-state / discrepancy / noise / drift calibration model*.

.. code-block:: text

   Step 0  initialize, measure baseline, estimate noise, freeze hard limits
         +-------------------------------------------------------------+
         | Step 1  propose booster correction u_B + measurement action  |
         | Step 2  set booster geometry, record achieved geometry       |
         | Step 3  align antenna (x, y) for this booster state          |
         | Step 4  measure selected observable / boost factor           |
         | Step 5  jointly update theta, discrepancy, noise, drift      |
         | Step 6  rebuild optimizer-facing posterior predictive model  |
         | Step 7  stop or repeat --------------------------------------+
         +--- final high-fidelity validation + feasibility report

At a glance
-----------

* **Safe by construction** — damage-relevant constraints are enforced
  exactly before any candidate reaches hardware; they are never learned by
  failure (:doc:`user_guide/safety`).
* **Statistically honest** — every measurement carries an uncertainty;
  detector-state parameters are inferred *jointly* with a simulator
  discrepancy model, measurement noise and drift
  (:doc:`user_guide/statistics`).
* **Budget-aware** — the acquisition weighs expected improvement against
  measurement cost and can fall back to replication, re-baselining, cheap
  low-fidelity probes, or a stop recommendation
  (:doc:`algorithm/step1`).
* **Hardware-agnostic** — the loop only talks to
  :class:`~madmax_calibration.hardware.HardwareInterface`; a fully
  simulated detector is included for offline validation
  (:doc:`user_guide/hardware`).

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started
   example_walkthrough

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/architecture
   user_guide/configuration
   user_guide/physics
   user_guide/statistics
   user_guide/safety
   user_guide/hardware

.. toctree::
   :maxdepth: 2
   :caption: The algorithm, step by step

   algorithm/index

.. toctree::
   :maxdepth: 1
   :caption: Validation

   testing

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Design documents

   design/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
