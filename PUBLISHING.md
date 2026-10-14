# Publishing the course site

The rendered site is a plain folder of static files (`_output/`), so any
static host will serve it. This file describes the GitHub Pages route,
because it is free, it is a single command once set up, and it can later
be put behind a TU/e address without the URL mentioning GitHub.

What gets published: the landing page (`index.qmd`), 9 slide decks, 8
sets of notes, 8 exercise sheets, 8 notebook pages, the `.ipynb` and
`.py` files for download, the figures and the videos, plus a PDF of each
notes/exercise page. About **250 MB**, 1700-odd files.

---

## One-time setup

### 1. Create the repository on GitHub

Create an **empty public** repository -- no README, no .gitignore. Call
it something like `4SC000` or `optimal-control-course`; the name shows up
in the URL until a custom domain is attached.

> It has to be **public** on a free GitHub account: Pages only works for
> private repositories on Pro/Team/Enterprise plans. The site is
> therefore readable by anyone who has the link. `robots.txt` asks search
> engines not to index it, so it will not turn up in a Google search, but
> it is not protected. See "If it has to be non-public" below.

### 2. Put the sources on it

In Terminal, from this folder:

```bash
cd "~/Documents/AssociateProfessor/1 Optimal Control Course/course-quarto"
git init -b main
git add .
git commit -m "Course materials: units 1-8"
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```

`_output/` and `.quarto/` are in `.gitignore`, so this pushes the sources
only (about 190 MB, mostly figures).

### 3. Render and publish the site

```bash
./publish.sh
```

This runs `quarto render` and then `quarto publish gh-pages`, which
copies `_output/` onto a separate `gh-pages` branch and pushes it. The
first run asks for confirmation and prints the site URL.

### 4. Turn Pages on

On GitHub: **Settings -> Pages -> Build and deployment**, Source
"Deploy from a branch", branch `gh-pages`, folder `/ (root)`. Save.
(`quarto publish` tries to do this for you; check it anyway.)

A minute or two later the site is at

```
https://<your-username>.github.io/<repo>/
```

That is the link to send out for feedback.

---

## Afterwards: every update

```bash
./publish.sh            # re-render everything, then publish
./publish.sh --fast     # publish what is already in _output
```

and commit the sources as usual (`git add -A && git commit && git push`).

---

## Later: a TU/e address instead of github.io

This is the same arrangement as a personal site on a custom domain. The
files stay on GitHub; the address does not mention it.

1. **Pick the hostname**, e.g. `optimalcontrol.tue.nl` or
   `4sc000.tue.nl`.

2. **Ask TU/e IT for one DNS record** -- a `CNAME` for that hostname
   pointing at `<your-username>.github.io.` (note the trailing dot; the
   target is the *account* host, not the repository). For a subdomain
   that single record is all that is needed. An apex domain (`tue.nl`
   itself, which they will not give you) would instead need four `A`
   records.

3. **Tell GitHub about it**: Settings -> Pages -> Custom domain, type the
   hostname, Save. Then tick **Enforce HTTPS** once the certificate has
   been issued (a few minutes; GitHub gets a free Let's Encrypt one).

4. **Make the setting survive the next publish.** Entering the custom
   domain writes a file called `CNAME` at the root of the `gh-pages`
   branch -- and `quarto publish gh-pages` rewrites that branch, which
   deletes it, and the custom domain silently falls back to github.io.
   Fix it once, permanently, by keeping the file in the sources:

   ```bash
   echo "optimalcontrol.tue.nl" > CNAME
   ```

   and adding it to the `resources:` list in `_quarto.yml`:

   ```yaml
   resources:
     ...
     - "CNAME"
   ```

   Now every render puts `CNAME` into `_output/`, so every publish
   restores it.

If TU/e will not create a DNS record, the fallback is a page on the TU/e
site that links to (or iframes) the github.io URL -- the address bar then
shows GitHub once the visitor follows the link.

---

## If `quarto publish gh-pages` refuses

It is documented for website and book projects; this one is a `default`
project (nine reveal.js decks and a pile of html/pdf documents rather
than one Quarto website), so if the command complains, publish the
branch by hand instead -- it does exactly the same thing:

```bash
cd "~/Documents/AssociateProfessor/1 Optimal Control Course/course-quarto"
quarto render

# once: create an empty gh-pages branch
git checkout --orphan gh-pages && git rm -rf . && git commit --allow-empty -m "init gh-pages" && git push -u origin gh-pages && git checkout main

# every time: push _output onto it through a worktree
rm -rf /tmp/ghp && git worktree add /tmp/ghp gh-pages
rsync -a --delete --exclude .git _output/ /tmp/ghp/
touch /tmp/ghp/.nojekyll
git -C /tmp/ghp add -A && git -C /tmp/ghp commit -m "publish site" && git -C /tmp/ghp push
git worktree remove /tmp/ghp
```

Replace the body of `publish.sh` with those last five lines if you end
up on this route.

---

## Things worth knowing

- **Exercise solutions are in the published pages.** They sit in
  collapsed callouts, which means they are hidden but present in the
  HTML source. Fine for a reviewer; think about it before giving the
  link to a cohort mid-course.

- **`robots.txt` keeps the site out of search engines.** Delete the file
  and its line in `_quarto.yml` when the materials should be findable.

- **Deployments time out after 10 minutes.** At ~250 MB this should be
  comfortable, but if a deployment ever fails on time, the easy savings
  are `lectures/figures/lecture1v3/movie_writingpolicy.mp4` (22 MB, by
  far the largest single file -- re-encoding it with ffmpeg would bring
  it to a few MB) and the 422 PNG screenshots (116 MB in total, most of
  them re-compressible without a visible difference).

- **`lectures/figures/` holds ~31 MB of figures no source file refers
  to** -- `lecture1/`, `lecture1v2/` and most of `lecture2/pg*.png` are
  superseded. Quarto does not copy unreferenced files into `_output`, so
  they do not reach the site, but they are in the repository. Worth a
  clean-out at some point.

- **Other hosts, if ever needed.** `_output/` is self-contained, so
  Netlify, Cloudflare Pages and GitLab Pages all work by dropping the
  folder in. Netlify and Cloudflare can put a password on a site, which
  GitHub Pages cannot.

### If it has to be non-public

Options, roughly in order of effort:

- Keep the public repo, rely on `robots.txt` and an unlisted URL. This
  is the normal arrangement for course material and almost certainly
  enough here.
- GitHub Pro (a few euros a month) allows Pages from a private
  repository -- the *site* is still public, but the *sources* are not.
- Netlify or Cloudflare Pages with password protection, for a site that
  nobody can read without the password.
