"""Download operation module."""

import dataclasses
import hashlib
import logging
import os
import time
import urllib.parse

from etils import epath
import requests
import tqdm

from mlcroissant._src.core import constants
from mlcroissant._src.core.optional import deps
from mlcroissant._src.core.path import get_fullpath
from mlcroissant._src.core.path import Path
from mlcroissant._src.operation_graph.base_operation import Operation
from mlcroissant._src.structure_graph.base_node import Node
from mlcroissant._src.structure_graph.nodes.file_object import FileObject

_DOWNLOAD_CHUNK_SIZE = 1024
_GITHUB_GIT = "https://github.com"
_GITLAB_GIT = "https://gitlab.com"
_HUGGING_FACE_GIT = "https://huggingface.co/datasets"
_SUPPORTED_HOSTS = [
    _GITHUB_GIT,
    _GITLAB_GIT,
    _HUGGING_FACE_GIT,
]


def is_url(url: str) -> bool:
    """Tests whether a URL is valid.

    The current version is very loose and only supports the HTTP protocol.
    """
    return url.startswith("http://") or url.startswith("https://")


def get_hash(url: str) -> str:
    """Retrieves the sha256 hash of an URL."""
    return hashlib.sha256(url.encode()).hexdigest()


def get_download_filepath(node: Node, url: str) -> epath.Path:
    """Retrieves the download filepath of an URL."""
    if not is_url(url):
        assert url.startswith("data/"), (
            'Local file "{self.node.uid}" should point to a file within the data/'
            ' folder next to the JSON-LD Croissant file. But got: "{self.url}"'
        )
        filepath = node.folder / url
        assert filepath.exists(), (
            f'In node "{node.uid}", file "{url}" is either an invalid URL'
            " or an invalid path."
        )
        # No need to download local files
        return filepath
    hashed_url = get_hash(url)
    constants.DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    return constants.DOWNLOAD_PATH / f"croissant-{hashed_url}"


def insert_credentials(url: str, username: str | None, password: str | None) -> str:
    """Inserts credentials in the URL for an HTTP authentication.

    Args:
        url: The Git URL.
        username: The username to use for the HTTPS authentication.
        password: The password to use for the HTTPS authentication.

    Returns:
        The URL populated with the given credentials. For example:
        `https://{username}:{password}@github.com/account/repo`.
    """
    https = "https://"
    if not url.startswith(https):
        raise ValueError("use secured HTTPS protocol for {url}.")
    if username is None and password is None:
        return url
    if username is None or password is None:
        raise ValueError(
            f"please provide both {constants.CROISSANT_GIT_USERNAME} and"
            f" {constants.CROISSANT_GIT_PASSWORD}."
        )
    username = urllib.parse.quote(username)
    password = urllib.parse.quote(password)
    https_with_credentials = f"{https}{username}:{password}@"
    return url.replace(https, https_with_credentials)


def extract_git_info(full_url: str) -> tuple[str, str | None]:
    """Extracts git-related information.

    Args:
        full_url: The full git URL for an HTTP clone.

    Returns:
        A tuple containing the base URL of the repository to clone and the refs to
        checkout if any is provided.
    """
    full_url = os.fspath(full_url)
    if full_url.startswith(_GITHUB_GIT) or full_url.startswith(_GITLAB_GIT):
        return full_url, None
    if full_url.startswith(_HUGGING_FACE_GIT):
        splits = full_url.rsplit("/tree/refs%2F", 1)
        if len(splits) == 1:
            # There is no refs in the URL: returning None.
            return splits[0], None
        else:
            url, refs = splits
            return url, f"refs/{urllib.parse.unquote(refs)}"
    else:
        raise ValueError(
            f"unknown git host: {full_url}. Supported git hosts:"
            f" {_SUPPORTED_HOSTS}. Contact the Croissant team to support more hosts."
        )


@dataclasses.dataclass(frozen=True, repr=False)
class Download(Operation):
    """Downloads from a URL to the disk."""

    node: FileObject
    url: str

    def _download_from_http(self, filepath: epath.Path):
        response = requests.get(self.url, stream=True, timeout=10)
        total = int(response.headers.get("Content-Length", 0))
        with filepath.open("wb") as file, tqdm.tqdm(
            desc=f"Downloading {self.url}...",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                size = file.write(data)
                bar.update(size)

    def _download_from_git(self, filepath: epath.Path):
        username = os.environ.get(constants.CROISSANT_GIT_USERNAME)
        password = os.environ.get(constants.CROISSANT_GIT_PASSWORD)
        # GIT_LFS_SKIP_SMUDGE allows to not download git-lfs files by default. Those
        # files may be big, so we want to download them just-in-time (see `Read`):
        os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
        url, refs = extract_git_info(self.node.content_url)
        url = insert_credentials(url, username, password)
        repo = deps.git.Repo.clone_from(url, filepath)
        # Hugging Face uses https://git-scm.com/book/en/v2/Git-Internals-Git-References.
        if refs:
            branch_name = f"branch_{time.time()}"  # Branch name with no conflict
            repo.remote().fetch(f"{refs}:{branch_name}")
            repo.branches[branch_name].checkout()

    def __call__(self, *args) -> epath.Path:
        """See class' docstring."""
        del args  # unused
        filepath = get_download_filepath(self.node, self.url)
        if not filepath.exists():
            if self.node.encoding_format == constants.GIT_HTTPS_ENCODING_FORMAT:
                self._download_from_git(filepath)
            else:
                self._download_from_http(filepath)
        logging.info("File %s is downloaded to %s", self.url, os.fspath(filepath))
        return Path(
            filepath=filepath,
            fullpath=get_fullpath(filepath, constants.DOWNLOAD_PATH),
        )
