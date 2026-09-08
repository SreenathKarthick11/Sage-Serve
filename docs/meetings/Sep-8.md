# Minutes

> **Date:** September 8, 2026

## Agenda

Discuss the project scope, expectations, proposed approach, and submission deliverables.

## Discussions

- Went through the **3-page SageServe paper** and gained an understanding of how SageServe works.
[Paper](https://dl.acm.org/doi/10.1145/3801489.3806875)

- Discussed the expectations and deliverables of the project.

### Proposed Approach

1. Run the **SplitWise simulator** on the paper's production traces.
2. Reproduce **reactive vs. forecast-aware scaling** results for instance-hours and latency.
3. Implement **one extension**: a scheduling policy or forecasting model.
4. Evaluate its **cost impact** in terms of $/GPU-hour saved.

### Deliverables

- Reproduction of **instance-hours and latency** results.
- **One implemented extension**.
- **Cost-savings analysis** based on the paper's methodology.


## TODO

- [ ] Read the complete **SageServe paper (24 pages)** and understand the architecture, forecasting, optimization, and evaluation methodology.
  [SageServe - arXiv](https://arxiv.org/pdf/2502.14617)

- [ ] Read through the **SplitWise simulator repository** and understand how to run and modify it for the project.
  [SplitWise Simulator - GitHub](https://github.com/mutinifni/splitwise-sim)
