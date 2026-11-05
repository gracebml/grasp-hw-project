# Hugging Face Space (Docker SDK) of the GRAT demo. The code comes from the public GitHub repository at a pinned commit;
# the checkpoints come from a Hugging Face model repository at start-up (entrypoint.sh).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user

ARG REPO_URL=https://github.com/gracebml/grasp-hw-project
ARG REF=main
RUN git clone "$REPO_URL" app && cd app && git checkout "$REF" && git submodule update --init --recursive

# CLIP's setup.py needs pkg_resources: old setuptools and no build isolation
RUN pip install --no-cache-dir --user "setuptools<70" wheel \
 && pip install --no-cache-dir --user --no-build-isolation -r app/demo/requirements.txt "huggingface_hub>=0.24"

ENV GRAT_CKPT_ROOT=/home/user/checkpoints CKPT_REPO=__CKPT_REPO__
COPY --chown=user entrypoint.sh /home/user/entrypoint.sh
EXPOSE 7860
CMD ["bash", "/home/user/entrypoint.sh"]
