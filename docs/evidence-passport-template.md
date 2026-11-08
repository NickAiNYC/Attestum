# Evidence Passport — {{passport_id}}

> Fill-in template. Every slot is required and must be consumed at
> render time; the acceptance script refuses output containing
> unresolved markers. How to fill: see `templates/README.md`. The
> worked anonymized example lives in `templates/examples/`.

**Customer:** {{customer_code}} · **Site:** {{site_code}} · **Cell:** {{cell_id}}
**Prepared by:** {{prepared_by}} · **Issued (UTC):** {{issued_utc}} · **Schema:** {{schema_version}}
**Passport hash** (sha256 over canonical JSON of this document minus the hash itself): `{{passport_hash}}`

---

## 1. Cell and robot identification

- Task: {{task_name}}
- Robot: {{robot_make}} {{robot_model}} (serial code {{robot_serial_code}}), {{dof_count}} degrees of freedom

## 2. Software, firmware, and configuration versions

- Runtime: {{runtime_package}} {{runtime_version}}, harness hash `{{harness_hash}}`
- Controller firmware: {{controller_firmware}} · Vision firmware: {{vision_firmware}}
- Configuration sha256: `{{configuration_sha256}}` · Prompt registry sha256: `{{prompt_registry_sha256}}`

## 3. Data provenance and timestamps

{{provenance_rows}}

- Site timezone: {{site_timezone}}
- Clock sync: {{clock_sync_method}}
- Exported by: {{export_operator_code}}

## 4. Baseline measurement window

- Window: {{start_utc}} → {{end_utc}} ({{operating_hours}} operating hours)
- Interventions in window: {{intervention_count}} ({{intervention_rate_per_hour}} per operating hour)

## 5. Frozen evaluation conditions

- Task definition sha256: `{{task_definition_sha256}}`
- Comparison design: {{comparison_design}}
- Exclusion rule: {{exclusion_rule}}
- Measurement pipeline: {{measurement_pipeline}} · recorder at {{recorder_rate_hz}} Hz
- Evaluation freeze tag: `{{evaluation_freeze_tag}}`

## 6. Failure taxonomy

| Label | Baseline count | Definition |
| --- | --- | --- |
{{taxonomy_rows}}

- Taxonomy source: {{taxonomy_source}}

## 7. Evaluation methodology

- Design: {{design}}
- Primary metric: {{primary_metric}}
- Sample: {{sample_description}}
- Interval reference: {{interval_reference}}
- Gate reference: {{gate_reference}}
- Acceptance rule: {{acceptance_rule}}

## 8. Statistical uncertainty

- Observed: {{observed_successes}} of {{observed_trials}} matched events eliminated
- Point estimate: {{point_estimate_pct}}%
- Wilson 95% CI: [{{ci_low_pct}}%, {{ci_high_pct}}%]
- Method: {{ci_method}} (z = {{z_value}})
- Notes: {{notes}}

## 9. Safety approvals and human oversight

- Safety approval reference: {{safety_approval_ref}}
- Approver role: {{approver_role}}
- Human oversight: {{human_oversight_description}}
- Stop conditions: {{stop_conditions_rows}}
- Toolchain guard version: {{toolchain_guard_version}}
- Analysis-only attestation: {{analysis_only_attestation}}

## 10. Reproducible artifacts and hashes

| Artifact | Kind | sha256 | Bytes |
| --- | --- | --- | --- |
{{artifacts_rows}}

## 11. Improvement and regression results

{{outcome_rows}}

> Section 11 is a ternary, not a binary: exactly one of
> {verified improvement, clear negative, no-go} is ever set. A clear
> negative ("the problem is mechanical") is a legitimate, publishable
> outcome.

---

_Issued by {{company_name}} — {{contact_email}}. Verify this document
against its passport hash before relying on it._
