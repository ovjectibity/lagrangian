1. Worker instances need to have IAM setup: Currently needed for pulling ECR images
2. Ensure that the instance has the correct security group setup (for correct networking traffic allowance - ssh, TCP, WSS traffic)
3. Ensure that the security group being used has the correct IP set for inbound rules. This is needed for admin setup & the server machine communication (for SSH,TCP traffic etc.)