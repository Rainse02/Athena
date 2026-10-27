# Versioned software archiving with Zenodo

This repository is prepared as a software archive containing study code,
configuration, and work descriptions. Preparing this revision does not itself
publish a Zenodo record or assign a DOI.

## Metadata

`CITATION.cff` records the software title, authors from the associated manuscript,
repository URL, license, abstract, and keywords. It omits an unissued DOI,
release version, and release date. These fields should be updated to the
actual release metadata when preparing the archival release.

Zenodo supports `CITATION.cff`; if `.zenodo.json` is also present, Zenodo gives
that file precedence. This repository therefore uses `CITATION.cff` as its
single citation-metadata source. See Zenodo's
[metadata documentation](https://help.zenodo.org/docs/github/describe-software/)
and [CITATION.cff guide](https://help.zenodo.org/docs/github/describe-software/citation-file/).

## Archival procedure

1. Complete the code revision intended for publication and record its Git
   commit. The release tree should contain the intended source, configuration,
   license, and documentation. Inputs and native models remain separate.
2. Set the software version and release date in `CITATION.cff` to match the
   intended release. Record the actual author metadata and article reference
   available at that time.
3. Sign in to Zenodo with the repository owner's account, connect GitHub,
   synchronize the repository list, and enable `Rainse02/Athena`. See the
   [repository-enabling guide](https://help.zenodo.org/docs/github/enable-repository/).
4. Publish a GitHub release from the selected commit and its version tag.
   Zenodo ingests new releases from an enabled repository. A Git push alone
   does not perform this archival step. See the
   [release-archiving guide](https://help.zenodo.org/docs/github/archive-software/github-upload/).
5. Inspect the resulting Zenodo files, metadata, and DOI. Cite the software
   version used for the study. Add the issued DOI to the repository citation
   information after it is available; do not insert a guessed DOI.

A release snapshot contains the selected code tree. Earlier Git commits can
still contain old files: deleting files in a new commit does not erase those
commits. Rewriting history is unnecessary for a code-only release snapshot.

## Reproducibility statement

The software DOI will identify archived code and documentation. It will not
imply that measured images, processed data, or native simulation models were
deposited with it. A separate data availability statement should describe where
the required inputs can be obtained. The manuscript's availability statement
should use issued identifiers and describe the actual deposited scope.
