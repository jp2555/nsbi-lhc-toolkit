Constituent-Level Density Ratios (HH→bbττ + Sophon)
======================================================

This page describes the **hierarchical constituent density-ratio estimator** introduced
for the HH→bbττ κ_λ example. It extends the standard :doc:`density_ratio_training`
workflow to per-jet constituent clouds, replacing the flat feature vector with a
two-level representation: a per-jet encoder that processes the full particle cloud
followed by an event-level set-transformer over jet and object tokens.

The key hypothesis is that the sophon-ak4 Particle Transformer (ParT) pre-trained on
large jet datasets provides a better jet embedding starting point than a randomly
initialised encoder, leading to faster convergence and better data efficiency — the
central diagnostic of the Phase-1 study.


Architecture
------------

The model processes each event in three stages:

1. **Per-jet constituent encoding.** For each AK4 jet, the padded constituent cloud
   (shape ``[N_PART_MAX, F_PART]``) is passed through a jet encoder. Three encoder
   variants are supported, all sharing the same interface:

   - ``stub`` — a lightweight masked DeepSets encoder (no external deps, fast on CPU;
     used in tests and as the no-foundation baseline).
   - ``scratch`` — a Particle Transformer of the sophon-ak4 shape, randomly initialised
     (architecture control: same expressivity as the pretrained model, no pretraining).
   - ``sophon-ak4`` — the same ParT loaded from the released sophon-ak4 checkpoint
     (available on HuggingFace at ``jet-universe/sophon-ak4``), either frozen or
     fine-tuned.

   All three return a ``(B, embed_dim)`` embedding per jet, where ``embed_dim=64``.

2. **Event set-transformer.** The per-jet embeddings and per-event object tokens
   (τ-leptons, electrons, muons, MET) are concatenated into a single token sequence.
   A masked multi-head self-attention transformer (``EventSetTransformer``) pools this
   sequence into a scalar logit, respecting the per-token padding masks so that absent
   jets and objects do not affect the output.

3. **Binary density-ratio head.** The logit is passed through a sigmoid to produce a
   per-event score :math:`s(x) \in (0, 1)`, which is converted to the density ratio
   :math:`r(x) = s(x) / (1 - s(x))` matching the downstream JAX fit contract.


Shape conventions
-----------------

All shape constants are defined in
:class:`~nsbi_common_utils.lightning_tools.cloud_spec.CloudSpec` and exported as
``DEFAULT_SPEC``:

.. code-block:: python

   from nsbi_common_utils.lightning_tools.cloud_spec import DEFAULT_SPEC
   # DEFAULT_SPEC.n_jets_max == 4, .n_part_max == 64, .f_part == 8
   # DEFAULT_SPEC.n_obj_max == 6, .f_obj == 6, .embed_dim == 64

Per-constituent features (``F_PART=8``):
  ``[log_pt, Δη, Δφ, charge, is_electron, is_muon, is_photon, is_charged_hadron]``

  The sophon-ak4 integration (Task 11) extends this to the full checkpoint schema
  including track-impact-parameter features ``d0/dz(+err)`` and additional PID one-hots.

Per-object-token features (``F_OBJ=6``):
  ``[log_pt, η, sin_φ, cos_φ, mass_or_MET, type_id]``
  where ``type_id ∈ {0: τ_had, 1: electron, 2: muon, 3: MET}``.


Training
--------

The trainer is :class:`~nsbi_common_utils.training.particle_ratio_estimation.particle_density_ratio_trainer`,
the constituent-cloud analogue of the flat-feature
:class:`~nsbi_common_utils.training.density_ratio_estimation.density_ratio_trainer`.
It wraps :class:`~nsbi_common_utils.lightning_tools.hh_density_ratio_model.HHDensityRatioLightning`
and handles data splitting, ONNX export, and ensemble management.

.. code-block:: python

   import numpy as np
   from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer

   clouds = dict(np.load("saved_clouds/hh_lam1.npz"))
   # clouds must contain: parts, part_mask, jet_mask, obj, obj_mask, y, w

   tr = particle_density_ratio_trainer(
       clouds=clouds,
       sample_name=["signal", "background"],
       output_name="hh_vs_ttbar",
       path_to_models="models/",
       encoder_kind="sophon-ak4",   # or "stub" / "scratch"
       freeze_backbone=True,        # frozen-first strategy
   )
   history = tr.train(number_of_epochs=100, batch_size=512, learning_rate=1e-3)
   ratios = tr.evaluate_ratios(clouds)

NLO event weights (potentially negative) are handled via a **label-flip** strategy:
events with ``w < 0`` are trained with ``|w|`` and the label inverted, keeping the
weighted binary cross-entropy finite and unbiased for signed-weight samples.


Comparison controls
-------------------

The ``run_controls`` function in
``examples/HH_bbtautau_kappalambda_sophon/compare.py`` trains all three encoder
variants under a common protocol and returns per-control training histories.

.. list-table::
   :header-rows: 1

   * - Control key
     - Encoder
     - Backbone frozen?
     - Purpose
   * - ``stub_scratch``
     - StubJetEncoder (DeepSets)
     - No
     - Fast CPU baseline; no constituent-topology inductive bias
   * - ``stub_frozen``
     - StubJetEncoder
     - Yes
     - Sanity: frozen trivial encoder
   * - ``scratch``
     - ParT (random init)
     - No
     - Architecture control: same ParT shape, no pretraining
   * - ``sophon_frozen``
     - sophon-ak4 ParT
     - Yes
     - Foundation model, transferred without fine-tuning
   * - ``sophon_finetune``
     - sophon-ak4 ParT
     - No
     - Foundation model, end-to-end fine-tuned

The low-statistics ablation (``low_stat_ablation``) trains the chosen control on
subsampled fractions of the training set to measure data efficiency as a function of
available statistics.


Constituent cloud format and the Delphes converter
---------------------------------------------------

Real Delphes simulation output is converted to the padded-cloud `.npz` format by
``examples/HH_bbtautau_kappalambda_sophon/scripts/delphes_to_clouds.py``.

.. code-block:: bash

   python scripts/delphes_to_clouds.py \
       --input hh_bbtautau_sm.root --tree Delphes \
       --output saved_clouds/hh_lam1.npz

The converter reads ``Jet.*``, ``EFlow.*``, ``MissingET.*``, and ``Event.Weight``
branches. For Delphes trees without an explicit constituent-to-jet index branch, replace
the ``EFlow.JetIndex`` assignment in ``assign_constituents()`` with a ΔR-matching
algorithm. The b-tag flavour label (``Jet.Flavor``) and τ-identification object tokens
can be added for the full signal-region selection.


Evaluation and the JAX fit contract
-------------------------------------

After training, per-event density ratios are written on the **shared Asimov event
ordering** (all basis-process events concatenated in a fixed order) by
``scripts/eval_to_ratios.py``:

.. code-block:: bash

   python scripts/eval_to_ratios.py \
       --clouds saved_clouds/ --model models/model0.onnx \
       --output ratios/

The output ``ratio_<process>.npy`` and ``weights.npy`` files are consumed directly by
:class:`~nsbi_common_utils.models.sbi_parametric_model.sbi_parametric_model` without
any changes to the JAX fit. This is the same contract as the flat-feature workflow; the
hierarchical model is a drop-in replacement at the ratio-production step.


Where it fits in the pipeline
------------------------------

This constituent-level density ratio replaces Stage 2 (flat-feature training) for the
HH→bbττ κ_λ analysis. Stages 3 (evaluation → `.npy`) and 4 (JAX fit) are identical
to the standard workflow:

1. **Data conversion** — ``scripts/delphes_to_clouds.py`` → ``.npz`` per process.
2. **Training** — ``compare.py::run_controls`` or ``particle_density_ratio_trainer``
   directly → ONNX models.
3. **Evaluation** — ``scripts/eval_to_ratios.py`` → per-event ratio ``.npy`` on the
   Asimov ordering.
4. **Fitting (Phase 2)** — ``sbi_parametric_model`` consumes the ratio arrays alongside
   the κ_λ morphing coefficients to build the statistical model.

See :doc:`density_ratio_training` for the flat-feature training reference and
:doc:`workflow` for the full four-stage pipeline overview.
