# App Server Setup (Remote Training Worker)

This document explains how to configure the app server for remote training mode.

## Overview

Remote training is used when `training_controller.acceleration_mode` is **not** `AllOnServer`.
In that case, the server:
1. Launches an AWS EC2 spot instance.
2. Starts the Python trainer Docker container on it.
3. Connects to the worker over WebSocket.
4. Reuses the same worker instance for subsequent remote jobs.
5. Terminates the instance when the server receives `Ctrl+C`.

Env vars are read from the server process environment (for local setup, use `.aws-env`).

## Required Environment Variables

| Variable | Description | How to obtain |
|---|---|---|
| `REMOTE_WORKER_AMI_ID` | AMI used to launch worker EC2 instance. | AWS Console -> EC2 -> AMIs (copy `ami-...`) or `aws ec2 describe-images ...`. |
| `REMOTE_WORKER_IMAGE_URI` | Worker Docker image URI (typically ECR). | AWS Console -> ECR -> repository -> copy Image URI. |
| `REMOTE_WORKER_SHARED_SECRET` | Shared auth secret used by server and worker WebSocket auth. | Generate and store securely (example: `openssl rand -hex 32`). Use same value on server and worker. |

## Optional Environment Variables

| Variable | Description | How to obtain |
|---|---|---|
| `REMOTE_WORKER_AWS_REGION` | AWS region for EC2/ECR operations. Default in code: `us-east-1`. | Your deployment region (for example `us-east-1`). |
| `REMOTE_WORKER_SUBNET_ID` | Subnet where EC2 instance should launch. | AWS Console -> VPC -> Subnets (`subnet-...`). |
| `REMOTE_WORKER_SECURITY_GROUP_IDS` | Comma-separated security group IDs for worker instance. | AWS Console -> EC2 -> Security Groups (`sg-...`). |
| `REMOTE_WORKER_KEY_NAME` | EC2 key pair name (optional SSH access). | AWS Console -> EC2 -> Key Pairs. |
| `REMOTE_WORKER_IAM_INSTANCE_PROFILE_ARN` | IAM instance profile ARN attached to worker instance. Recommended for ECR pull permissions. | AWS Console -> IAM -> Roles / Instance profiles (copy ARN). |
| `REMOTE_WORKER_INSTANCE_NAME_TAG` | Name tag for launched instance. | Any string value you choose. |
| `REMOTE_WORKER_SPOT_MAX_PRICE_USD` | Spot max bid price (USD as string). | Chosen by cost policy (example: `0.40`). |
| `REMOTE_WORKER_CONTAINER_NAME` | Docker container name on EC2 worker. | Any string value you choose. |
| `REMOTE_WORKER_PORT` | Worker container/listener port. Must match worker app config. | Default worker port is `8765` unless changed. |
| `REMOTE_WORKER_WS_PATH` | Worker WebSocket path. | Default `/` unless worker route changed. |

## AWS SDK Credentials (Server-side)

Set these only if your server environment does not already have valid AWS credentials (for example via profile, SSO, or instance role):

| Variable | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key for server process. |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for server process. |
| `AWS_SESSION_TOKEN` | Session token (if using temporary credentials). |
| `AWS_DEFAULT_REGION` | Default AWS region fallback for SDK. |

## Minimum AWS Permissions

The server principal (the credentials used by the Rust app) should be able to:
- Launch and terminate EC2 instances.
- Describe EC2 instances.
- Use networking resources selected (subnet, security groups).
- Pass instance profile if `REMOTE_WORKER_IAM_INSTANCE_PROFILE_ARN` is set (`iam:PassRole` / pass instance profile).

The launched worker instance role should be able to:
- Pull from ECR if image is private (`AmazonEC2ContainerRegistryReadOnly` or equivalent custom policy).

## Networking Requirements

- Security group should allow inbound traffic on `REMOTE_WORKER_PORT` from the app server.
- Worker instance needs outbound access to install packages and pull Docker images.
- If using private subnets, ensure NAT / routing is configured.

## Local Setup Flow

1. Fill values in `.aws-env` at repository root.
2. Load env vars into shell before starting server:
   - `source .aws-env`
   - or use `.envrc`/`direnv`.
3. Start app server normally.
4. Trigger a training job with acceleration mode other than `AllOnServer`.

## Recommended First Test

1. Use a known-good AMI and subnet.
2. Set a valid ECR image URI.
3. Set a new random shared secret in both server and worker env.
4. Start one short training run in remote mode and verify:
   - EC2 instance launches.
   - Worker container starts.
   - Training updates are streamed back.
   - Instance persists after job completion.
5. Stop server with `Ctrl+C` and verify instance termination.
