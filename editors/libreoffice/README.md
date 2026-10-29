# langcheck for LibreOffice

A grammar checker for Writer that shows suggestions from a shared
[langcheck](https://github.com/tallen12/langcheck) server as underlines with
suggestions. Paragraphs are answered from a cache at once; the whole document goes to
the server in the background, and Writer re-checks when new suggestions arrive.

## Install

1. Build `dist/langcheck.oxt` with `just libreoffice`, then install it in
   **Tools > Extension Manager** and restart LibreOffice.
2. Tell it where the server is, either:
   - set `LANGCHECK_SERVER__URL` and `LANGCHECK_SERVER__API_KEY` in the environment
     LibreOffice starts in, or
   - open **Tools > Options > LibreOffice > Advanced > Open Expert Configuration**,
     search for `org.langcheck`, and set `Url` and `ApiKey`.
3. Check **Tools > Options > Language Settings > Writing Aids** lists langcheck and
   **Tools > Automatic Spell Checking** is on.

English locales (US, GB, AU, CA, NZ) use langcheck instead of the built-in grammar
checker.
