
serve: dependencies
	bundle exec jekyll serve & \
	xdg-open http://localhost:4000 ; \
	wait

dependencies:
	 bundle install --path vendor/bundle
