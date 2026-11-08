# Attestum — Strategy Summary

An executive summary of the North Star document. One line carries the weight of the whole strategy:

> The company is built by producing two independently verified results where the second costs less than the first. Everything else is secondary.

## The product

We sell verified evidence, not robotics. The product is the Evidence Passport: an eleven-section, hash-verifiable record that a specific robot cell in a specific plant improved by a measured amount, under frozen conditions, with the uncertainty attached. Everything the company does either produces a passport or reduces the cost of producing the next one. Anything that does neither is out of scope.

## The wedge

The retrospective assessment. We need no live access to the customer's cell, no safety review, and no change to production: the customer ships data they already have (controller logs, shop-floor video, intervention records), and twenty-one days after signing they hold a passport, an economic case, and a case-study draft. The timeline is fixed — Day 0 signed, Day 5 data received, Day 15 draft, Day 20 review, Day 21 final — and it is fixed because a bounded engagement is the only one a plant manager can approve without a committee. Because we never touch the line, the engagement is safe to buy, cheap to deliver, and repeatable — the assessment is analysis, not integration.

## The economics

The assessment is free when the case study is published; withdrawal before publication converts the fee retroactively to $20,000 USD. This is not a discount scheme — it is the qualification mechanism. The free path is taken only by customers willing to stand behind published numbers, which is precisely the set whose passports build the company's reputation. The case study is not marketing garnish; it is the second product of every engagement, and the reason the next customer takes the first meeting.

## The engine

The load-bearing line, operationalized: assessment one costs engineering time; assessment two must cost less, because the templates, the renderer, the replay pipeline, and the taxonomy mappings are already built. Every engagement hardens the machinery for the next one. If the second assessment cannot be produced for less than the first, the company is consulting, not compounding — and the strategy has failed.

## The open-source posture

The runtime (robocore) is open source because openness is distribution, not charity: a public, hash-chained, property-fuzzed measurement runtime is the reason a stranger should believe a passport. The runtime is not the product and never will be; it is the trust substrate under the product. The runtime keeps its own package name; the retired umbrella name survives, if at all, only as the open-source distribution channel for the runtime — never as the company or the repository.

## The discipline

Until the first signed assessment is delivered and published, we build no operational infrastructure: no CLI, no dashboards, no custom hardware support, no multi-industry expansion, no additional cognitive machinery, no claim that we own a customer's improvement. The deferred list lives in docs/roadmap.md with a reason on every line. Building infrastructure before the first signed customer is the single largest failure mode available to this project, and we treat it that way in writing so it is checkable.

## The integrity

The passport reports a ternary: verified improvement, clear negative, or no-go. A clear negative is a publishable, chargeable outcome — "the problem is mechanical" saves the customer more money than a manufactured improvement ever could. We do not promise performance improvements, autonomous operation, or machinery safety certification, and the passport says so in writing. Trust is the only asset this company compounds; the ternary is how it stays bankable.
