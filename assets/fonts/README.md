Bundled HUD fonts

- `PlusJakartaSans-VariableFont_wght.ttf`
- `BebasNeue-Regular.ttf`
- `Onest-Regular.ttf`
- `Onest-Medium.ttf`
- `Onest-SemiBold.ttf`

The app loads these repo-local fonts first so a fresh GitHub checkout renders the intended HUD typography without requiring system font installation.

`Neue Haas Grotesk` is not bundled here. If a user already has it installed locally, the HUD loader may use it as an optional fallback for a few compact UI roles.
