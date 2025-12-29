> [!NOTE]
> This package is still under development. Always use the latest version for better stability.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <a href="https://github.com/llamp-ai/llamphouse">
    <img src="docs/img/llamphouse.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">LLAMPHouse</h3>

<p align="center">
    Serving Your LLM Apps, Scalable and Reliable.
    <br />
    <a href="https://github.com/llamp-ai/llamphouse/tree/main/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/llamp-ai/llamphouse">View Demo</a> -->
    ·
    <a href="https://github.com/llamp-ai/llamphouse/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/llamp-ai/llamphouse/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- PROJECT DESCRIPTION -->

# Introduction

Building LLM-powered applications is easier than ever, with countless frameworks helping you craft intelligent workflows in Python. But when it’s time to deploy at scale, the challenges begin.

Most tutorials suggest spinning up a FastAPI server with an endpoint — but what happens when scalability and reliability  becomes critical?

**That’s where LLAMPHouse comes in.**

LLAMPHouse provides a self-hosted, production-ready server that mimics OpenAI’s Assistant API while giving you full control over execution. Whether you're using LangChain, LlamaIndex, or your own custom framework, LLAMPHouse lets you deploy, scale, and customize your LLM apps—without sacrificing flexibility.

![assistant API](docs/img/assistant_api.png)

Take control of your LLM infrastructure and build AI-powered apps on your own terms with LLAMPHouse. 🚀

<!-- GETTING STARTED -->

## Getting Started

Requires Python 3.10+.

```
pip install llamphouse
```

LLAMPHouse uses an in-memory data store by default (no database required).
To enable Postgres, set:

```
DATABASE_URL="postgresql://postgres:password@localhost/llamphouse"
```

<!-- USAGE EXAMPLES -->

## Usage

LLAMPHouse supports pluggable backends:

- data_store: in_memory (default) or postgres
- event queue: in_memory or janus

_Streaming adapters are available for OpenAI, Gemini, and Anthropic.
See [Examples](examples/) for full runnable samples._

<!-- DEVELOPMENT -->

## Development

### Local

1. Clone the repository
2. Install the library `pip install .`

### Build

This is only required if you want to push the package to PyPI.

1. `python setup.py sdist bdist_wheel`
2. `git tag -a v1.0.0 -m "Release version 1.0.0"`
3. `git push`

### Testing

1. Install the package locally.
2. Run tests:

   ```bash
   python -m pytest tests/unit tests/contract tests/integration
   ```
3. Optional Postgres tests:

   - set `DATABASE_URL` and run:

     ```bash
     python -m pytest -m postgres
     ```

### Database (Postgres only)

Use Alembic when running the postgres data_store:

1. ```bash
   docker run --rm -d --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -p 5432:5432 postgres
   ```
2. ```bash
   docker exec -it postgres psql -U postgres -c 'CREATE DATABASE llamphouse;'
   ```

To create a new database revision: `alembic revision --autogenerate -m "Added account table"`

To upgrade the database with the latest revision: `alembic upgrade head`

To downgrade back to the base version: `alembic downgrade base`

<!-- ENDPOINTS -->

## Included API endpoints

- Assistants

  - ~~Create~~  ->  created in code
  - [X] List
  - [X] Retrieve
  - ~~Modify~~  ->  only in code
  - ~~Delete~~  ->  only in code
- Threads

  - [X] Create
  - [X] Retrieve
  - [X] Modify
  - [X] Delete
- Messages

  - [X] Create
  - [X] List
  - [X] Retrieve
  - [X] Modify
  - [X] Delete
- Runs

  - [X] Create
  - [X] Create thread and run
  - [X] List
  - [X] Retrieve
  - [X] Modify
  - [X] Submit tool outputs
  - [X] Cancel
- Run steps

  - [X] List
  - [X] Retrieve
- Vector stores

  - [ ] Create  ->  depends on implementation
  - [ ] List
  - [ ] Retrieve
  - [ ] Modify
  - [ ] Delete  ->  depends on implementation
- Vector store files

  - [ ] Create
  - [ ] List
  - [ ] Retrieve
  - [ ] Delete
- Vector store file batches

  - [ ] Create
  - [ ] Retrieve
  - [ ] Cancel
  - [ ] List
- Streaming

  - [X] Message delta
  - [X] Run step object
  - [X] Assistant stream

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/llamp-ai/llamphouse/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=llamp-ai/llamphouse" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->

## License

See [`LICENSE`](LICENSE) for more information.

<!-- CONTACT -->

## Contact

Project Admin: Pieter van der Deen - [email](mailto:pieter@llamp.ai)

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/llamp-ai/llamphouse?style=for-the-badge
[contributors-url]: https://github.com/llamp-ai/llamphouse/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/llamp-ai/llamphouse?style=for-the-badge
[forks-url]: https://github.com/llamp-ai/llamphouse/network/members
[stars-shield]: https://img.shields.io/github/stars/llamp-ai/llamphouse.svg?style=for-the-badge
[stars-url]: https://github.com/llamp-ai/llamphouse/stargazers
[issues-shield]: https://img.shields.io/github/issues/llamp-ai/llamphouse.svg?style=for-the-badge
[issues-url]: https://github.com/llamp-ai/llamphouse/issues
[license-shield]: https://img.shields.io/github/license/llamp-ai/llamphouse.svg?style=for-the-badge
[license-url]: https://github.com/llamp-ai/llamphouse/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/pieter-vdd
