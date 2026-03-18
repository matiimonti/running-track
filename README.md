 Check python version and create Venv
 
```
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```
## API Key Rotation

All secrets are loaded from `.env` and never hardcoded. To rotate a key:

### Strava
1. Go to https://www.strava.com/settings/api
2. Reset the client secret
3. Update `STRAVA_CLIENT_SECRET` in your deployment environment variables
4. Redeploy — no code change needed

### Claude
1. Go to https://console.anthropic.com
2. Revoke the old key, generate a new one
3. Update `CLAUDE_API_KEY` in your deployment environment variables
4. Redeploy

### Mapbox
1. Go to https://account.mapbox.com/access-tokens
2. Delete the old token, create a new one with the same scopes
3. Update `MAPBOX_TOKEN` in your deployment environment variables
4. Redeploy

> Never commit `.env` to git. Rotate immediately if a key is ever exposed in a commit.
