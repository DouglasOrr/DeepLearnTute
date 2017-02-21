# Deployment Notes

On the Azure web UI:

 - Machine: Docker on Ubuntu Server
 - Login: ssh
 - Pricing: Standard G1
 - Endpoints: Add HTTP (TCP) 80:80
 - Resource group: deep-learn-tute
 - Location: West Europe
 - Pin to dashboard

Wait for it to start, SSH into machine, then:

    wget https://raw.githubusercontent.com/DouglasOrr/DeepLearnTute/master/scripts/deploy
    chmod +x deploy
    ./deploy prepare USER1 USER2...
    ./deploy start
