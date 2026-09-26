# Minutes

> **Date:** September 22, 2026

## Agenda

- Discuss the 22 page paper
- Decide upon what are the target for midsem presentation.
- Divide tasks for the presentation.

## Target for Midsem Presentation

- [x] Clone the Sage Server repo, into this repository.
- [ ] Make slides summerising the paper.
- [ ] Add a slide explain our goal for the project.
- [ ] Run the SageServe Program in Conda : [Repo Link](https://github.com/shashwatj07/SageServe)
- [x] Google Slides is used for the presentation. [Google Slide Link](https://docs.google.com/presentation/d/1nEfR7WGiZX0B1zEJosX6CK1tyWVI0BmDf5KjZJzLkqY/edit?slide=id.p#slide=id.p)
- [x] Answers for the below [questions](#questions)

> [!Note]
> The SageServe Repo is a clone of the Splitwise repo , which has been modified.

## Task Division

- Aakash   : setup and run originial Spiltwise Repo.
- Akshat   : make slides from section (section 5)
- Simeon   : read and understand the Splitwise code base.
- Sreenath : make the intial slides.

## Questions

1. How does an LLM classify IW/NIW **before** it decides GPU allocate accordingly ? How much overhead does it causes ?

> **ANSWER**  \
> There is actually no LLM-based IW/NIW classification in SageServe. The workload tier is part of the request's SLA metadata, so classification doesn't require another model inference. The measured control-plane overhead is about 0.7 seconds for ARIMA forecasting and 1.5 seconds for ILP optimization, and these run hourly rather than per request

2. See how, useful ARIMA is for forcasting , is it accurate enough, is the speed-accuracy trade-off worth it ?

> **ANSWER** \
> “HMM and TVAR can provide better prediction accuracy, but they take considerably longer to train. SageServe doesn't need perfect prediction; it needs sufficiently accurate prediction for hourly GPU provisioning. ARIMA adds only about 0.7 seconds of forecasting overhead, so the authors choose it as a practical accuracy–latency trade-off. They also add a buffer and use reactive scaling to handle prediction errors.”

3. Where do we get the values for ILP optimization problem ?

> **ANSWER**
>
> | ILP parameter | Meaning | Where it comes from |
> |---|---|---|
> | $n_{i,j,k}$ | Current number of model instances | Current infrastructure state |
> | $\rho_{i,j}(w)$ | Forecasted TPS demand | **ARIMA** |
> | $\theta_{i,k}$ | TPS capacity of model $i$ on GPU $k$ | **Benchmarking** |
> | $\alpha_k$ | Cost of acquiring GPU VM $k$ | Public/cloud VM pricing |
> | $\sigma_{i,k}$ | Cost of starting model $i$ on GPU $k$ | GPU cost × measured startup time |
> | $\delta_{i,j,k}$ | Number of instances to add/remove | **ILP output** |
>

4. How does **Request Scheduling Polices : DPA** know when a task's (like NIW) deadline is ?

> **ANSWER** \
> DPA gets the deadline from the request’s SLA/deadline information; it does not predict the deadline itself. For each request, it calculates the remaining time until its deadline and uses that to determine its priority. In the paper’s experiments, NIW requests have a 24-hour deadline, so DPA tracks how long each NIW request has been waiting and increases its priority as it approaches or passes the deadline.