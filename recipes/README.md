# Sonar model recipes

This directory carries upstream deployment recipes adapted to Sonar's
`sonar` executable, Python namespace, and `SONAR_*` environment variables.
There is one YAML per model at `models/<hf_org>/<hf_repo>.yaml`; the path mirrors
the real Hugging Face model ID.

The upstream site, command builder, JSON API, schema, and shared vocabulary
(hardware profiles, serving strategies, and KV-offload deployments) are owned by
[vllm-project/recipes](https://github.com/vllm-project/recipes). Published model
IDs, container images, reference URLs, and upstream schema fields such as
`min_vllm_version` retain their real names. That version field describes the
upstream requirement, not a Sonar release number.

Pinned upstream and vendor images are external deployment references, not Sonar
images. They contain upstream executables and use upstream environment names;
use a Sonar installation when running the translated commands in these recipes.
Do not install Sonar from an upstream vLLM wheel index or assume an upstream
container tag is published for Sonar. See the
[Sonar installation guide](https://sonar.dphn.ai/getting-started/installation/)
for supported Sonar packages and containers.

Sonar package commands use the platform indexes at
`https://sonar.dphn.ai/whl/<release|nightly>/<backend>/<architecture>/simple`.
CUDA commands select the `cu130` PyTorch backend. For CPU packages, use the
installer so it also installs the matching Triton CPU dependency.

Commands that invoke `dynamo.vllm` or an external router belong to those
third-party runtimes. This rename does not add a Sonar backend to them.
Their published images and upstream launch flags are reference material, not
validated Sonar integrations.

Omni recipes remain in the upstream recipes repository and are not part of
Sonar's runtime. References to `vllm-omni`, `vllm-ascend`, and TPU integrations
identify separate upstream projects.

## Editing a recipe

Start a new recipe from
[`templates/recipe.yaml`](https://github.com/vllm-project/recipes/blob/main/templates/recipe.yaml).
Fields are defined in
[SCHEMA.md](https://github.com/vllm-project/recipes/blob/main/SCHEMA.md) and
checked against [`recipe.schema.json`](https://recipes.vllm.ai/recipe.schema.json)
on every build; an unknown or misspelled key is an error. For completion and
inline errors in YAML-aware editors, start the file with
`# yaml-language-server: $schema=https://recipes.vllm.ai/recipe.schema.json`.

Follow the recipes repository's
[add-recipe skill](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md)
for the schema and the validation steps. To validate schema compatibility, run
from a checkout of `vllm-project/recipes`, using its existing upstream source key:

```bash
node scripts/validate-source.mjs --source vllm --dir /path/to/sonar/recipes
```

To preview these files in that checkout, add a gitignored `sources.local.yaml`
with `overrides: { vllm: /path/to/sonar/recipes }` and run `pnpm dev`. This does
not register a new Sonar source with the upstream site.
