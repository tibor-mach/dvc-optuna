terraform {
  required_providers { iterative = { source = "iterative/iterative" } }
}
provider "iterative" {}

resource "iterative_task" "tpi_testing" {
  cloud      = "aws" # or any of: gcp, az, k8s
  machine    = "s"   # medium. Or any of: l, xl, m+k80, xl+v100, ...
  spot       = 0     # auto-price. Default -1 to disable, or >0 for hourly USD limit
  disk_size  = -1    # GB. Default -1 for automatic
  region     = "eu-central-1"
  timeout    = 3600 # terminate after 1 hour

  storage {
    workdir = "."       # default blank (don't upload)
    output  = "results" # default blank (don't download). Relative to workdir
  }
  script = <<-END
    #!/bin/bash
    sudo apt-get update \
    && sudo apt-get -y upgrade \
    && sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python-is-python3 \
    && sudo rm -rf /var/lib/apt/lists/*

    python -m pip install -r requirements.txt

    # create output directory if needed
    mkdir -p results
    echo "$GREETING" | tee results/$(uuidgen)
    # read last result (in case of spot/preemptible instance recovery)
    if test -f results/epoch.txt; then EPOCH="$(cat results/epoch.txt)"; fi
    EPOCH=$${EPOCH:-1}  # start from 1 if last result not found

    echo "(re)starting training loop from $EPOCH up to 100 epochs"
    for epoch in $(seq $EPOCH 100); do
      sleep 1
      echo "$epoch" | tee results/epoch.txt
    done
  END
}

output "logs" {
  value = try(join("\n", iterative_task.tpi_testing.logs), "")
}

output "events" {
  value = try(join("\n", iterative_task.tpi_testing.events), "")
}

output "status" {
  value = try(join("\n", iterative_task.tpi_testing.status), "")
}
