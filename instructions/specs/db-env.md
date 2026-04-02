# DB Environment Variables

All variables have defaults set in `scripts/init_db.sh` so the script
runs with no configuration on a standard local setup. Override any of
them in your shell or `.env` file before running the script.

---

## Postgres

| Variable            | Default       | Used by                        |
|---------------------|---------------|--------------------------------|
| `POSTGRES_HOST`     | `localhost`   | `init_db.sh`                   |
| `POSTGRES_PORT`     | `5432`        | `init_db.sh`                   |
| `POSTGRES_USER`     | `lagrangian`  | `init_db.sh`                   |
| `POSTGRES_PASSWORD` | `lagrangian`  | `init_db.sh`                   |
| `POSTGRES_DB`       | `lagrangian`  | `init_db.sh`                   |
| `DATABASE_URL`      | —             | app server (`main.rs`)         |

`DATABASE_URL` is what the app server reads at runtime. Format:

```
DATABASE_URL=postgres://<user>:<password>@<host>:<port>/<db>
```

Example (matching the defaults above):

```
DATABASE_URL=postgres://lagrangian:lagrangian@localhost:5432/lagrangian
```

---

## MongoDB

| Variable       | Default       | Used by                        |
|----------------|---------------|--------------------------------|
| `MONGODB_HOST` | `localhost`   | `init_db.sh`                   |
| `MONGODB_PORT` | `27017`       | `init_db.sh`                   |
| `MONGODB_DB`   | `lagrangian`  | `init_db.sh`                   |
| `MONGODB_URI`  | —             | app server (`main.rs`)         |

`MONGODB_URI` is what the app server reads at runtime. Format:

```
MONGODB_URI=mongodb://<host>:<port>
```

Example (matching the defaults above):

```
MONGODB_URI=mongodb://localhost:27017
```

---

## App server

| Variable         | Required | Description                              |
|------------------|----------|------------------------------------------|
| `DATABASE_URL`   | No       | Postgres connection string (see above).  |
| `MONGODB_URI`    | No       | MongoDB connection URI (see above).      |
| `CLIENT_DIST_DIR`| Yes      | Path to the built Svelte client (`dist/`).|

Both DB vars are optional at startup — the server will log a warning
and continue without DB support if they are unset.
