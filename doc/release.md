# Releasing notes

    # 1. Ensure code is committed & pushed (manual)

    # 2. Publish Docker image
    ./scripts/build
    docker tag deep-learn-tute douglasorr/deep-learn-tute:VERSION
    docker tag deep-learn-tute douglasorr/deep-learn-tute:latest
    docker push douglasorr/deep-learn-tute:VERSION
    docker push douglasorr/deep-learn-tute:latest

    # 3. Tag the release
    git push origin HEAD:refs/tags/VERSION

    # 4. Increment version number in setup.py (manual), commit & push
