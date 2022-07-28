#Update Board
sudo apt-get upgrade
sudo apt-get update

#Install Pip
sudo apt-get install python3-pip

#Install OpenCV
sudo apt-get install python3-opencv

#Install cscore
echo 'deb http://download.opensuse.org/repositories/home:/auscompgeek:/robotpy/Debian_10/ /' | sudo tee /etc/apt/sources.list.d/home:auscompgeek:robotpy.list
curl -fsSL https://download.opensuse.org/repositories/home:auscompgeek:robotpy/Debian_10/Release.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/home_auscompgeek_robotpy.gpg > /dev/null
sudo apt update
sudo apt install python3-cscore