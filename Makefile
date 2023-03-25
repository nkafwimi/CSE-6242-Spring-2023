GOOGLE_ADC_JSON = .config/gcloud/application_default_credentials.json
CURRENT_USER = $(shell whoami | tr '.' '_' )  # label cannot contain dot character


help:  ## Get a description of what each command does
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


build:  ## Build docker image locally for development
	docker build . -t $(IMAGE):latest --build-arg _=$(shell cp -R ../utils ./src)


push:  ## Push image to registry
	docker tag $(IMAGE) $(REGISTRY)/$(IMAGE)
	docker push $(REGISTRY)/$(IMAGE):latest

local-push:  ## Push image to registry, while on local
	docker tag $(IMAGE) $(LOCAL_REGISTRY)/$(IMAGE)
	docker push $(LOCAL_REGISTRY)/$(IMAGE):latest

run-container:  ## Run container
	docker run --rm -it \
	-v ~/$(GOOGLE_ADC_JSON):/tmp/gcp_credentials.json:ro \
	-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_credentials.json \
	$(IMAGE)


run-container-bash:  ## Run bash inside container
	docker run --rm -it \
	--entrypoint="/bin/bash" \
	-v ~/$(GOOGLE_ADC_JSON):/tmp/gcp_credentials.json:ro \
	-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_credentials.json \
	$(IMAGE)
