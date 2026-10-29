# Skills

Shared agent skills for this repository. Each skill is a folder:

```text
.agents/skills/<skill-name>/
├── SKILL.md        # front matter (name, description) + instructions
├── scripts/        # optional, run with `uv run`
├── references/     # optional supporting docs
└── assets/         # optional templates
```

`.claude/skills` is a symlink to this folder, so Claude Code loads the same skills. Follow
the rules in `AGENTS.md` ("Working across agents").
