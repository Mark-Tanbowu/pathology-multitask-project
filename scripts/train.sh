#!/usr/bin/env bash
set -e
python -m src.engine.train +experiment=model_unet data.root=data/processed trainer.max_epochs=2
