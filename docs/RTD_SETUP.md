# Setting Up Read the Docs

Quick guide to get your documentation published on Read the Docs.

## Step 1: Sign Up & Import

1. Go to https://readthedocs.org
2. Click "Sign Up" and use "Sign in with GitHub"
3. Authorize Read the Docs to access your GitHub account
4. Click "Import a Project"
5. Find and select `barisozmen/deepaugment`
6. Click "Next" (all defaults are fine)

## Step 2: Wait for Build

RTD will automatically:
- Detect `docs/conf.py`
- Install dependencies using `.readthedocs.yaml`
- Build your documentation
- Publish to `https://deepaugment.readthedocs.io`

First build takes ~2-5 minutes.

## Step 3: Add Badge to README

Once published, add this badge to your README.md:

```markdown
[![Documentation Status](https://readthedocs.org/projects/deepaugment/badge/?version=latest)](https://deepaugment.readthedocs.io/en/latest/?badge=latest)
```

It will show build status and link to your docs!

## Step 4: Configure (Optional)

### Custom Domain

In RTD dashboard:
1. Admin → Domains
2. Add your domain (e.g., `docs.deepaugment.com`)
3. Follow DNS setup instructions

### Versions

RTD automatically creates docs for:
- Latest commit (latest)
- All git tags (v1.0.0, v2.0.0, etc.)
- All branches (if enabled)

Users can switch versions in the docs!

### Email Notifications

Get notified of build failures:
1. Admin → Notifications
2. Add your email

### Analytics

See visitor stats:
1. Admin → Analytics
2. View traffic, search queries, popular pages

## Troubleshooting

### Build Failed

Check build logs in RTD dashboard. Common issues:

**Missing dependencies:**
- Add them to `.readthedocs.yaml` under `post_install`

**Import errors:**
- Make sure `docs/conf.py` has correct `sys.path`
- Check that package installs correctly with `pip install .`

**Warnings as errors:**
- Set `fail_on_warning: false` in `.readthedocs.yaml`

### Old Version Showing

- RTD caches builds
- Go to "Builds" tab and click "Build Version" to rebuild
- Or push a new commit

### Version Not Showing

- Check Admin → Versions
- Activate the versions you want public

## Advanced Features

### Pull Request Previews

RTD can build docs for PRs:
1. Admin → Advanced Settings
2. Enable "Build pull requests for this project"

Now every PR gets a preview link!

### Subprojects

Link multiple doc sets:
1. Admin → Subprojects
2. Add related projects

Example: Main project + plugins all searchable together

### Redirects

Set up URL redirects:
1. Admin → Redirects
2. Add redirect rules

Useful when restructuring docs.

## What You Get

Once set up, your docs will have:

✅ Beautiful URL: `https://deepaugment.readthedocs.io`
✅ Multiple formats: HTML, PDF, EPUB
✅ Version selector: Switch between v1, v2, latest, etc.
✅ Search: Full-text search across all docs
✅ Mobile-friendly: Responsive design (thanks to Furo!)
✅ Dark mode: Built into Furo theme
✅ Auto-updates: Rebuilds on every push
✅ SSL/HTTPS: Secure by default
✅ Fast CDN: Served globally
✅ Analytics: See what users read

## Example Projects on RTD

See how others do it:
- https://requests.readthedocs.io
- https://django.readthedocs.io
- https://pytorch.org (uses RTD backend)
- https://flask.readthedocs.io
- https://numpy.org/doc

## Support

- RTD Docs: https://docs.readthedocs.io
- RTD Support: support@readthedocs.org
- Community: https://github.com/readthedocs/readthedocs.org/issues

---

**Total Time**: ~5 minutes for initial setup, then automatic forever! 🎉
