docker run -d \
       --name gang_team \
       -e PORT=20160 \
       -p 20160:20160 \
       -e base_url=http://host.docker.internal:20166 \
       --add-host host.docker.internal:host-gateway \
       gang_team:v1.0