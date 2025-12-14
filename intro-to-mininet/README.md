# Mininet Setup Tips

<!---
My instance is ```thriving-crayfish```
--->

I am using the [Canonical multipass](https://canonical.com/) to create a light-weight linux instance. If you prefer aother VM setup, that is totally fine.

The ubuntu version should be 22.04 LTS. (Mininet does not support later versions.)

On the Ubuntu instance, you need to
1. install Mininet
2. install Open vSwitch (This is due to some version conflicts with the current Mininet)

## 1. Install Mininet
I use [the Native Installation from Source](https://mininet.org/download/#option-2-native-installation-from-source).

I am using Mininet **2.3.1b4**. 
```bash
mininet/util/install.sh -a
```

## 2. Install Open vSwitch
```bash
apt update
apt install openvswitch-switch
```

## Test your installations
You can test the environment with the following command:
```bash
sudo mn --switch ovsbr --test pingall
```

You should see something like this:
```bash
*** Creating network
*** Adding controller
*** Adding hosts:
h1 h2
*** Adding switches:
s1
*** Adding links:
(h1, s1) (h2, s1)
*** Configuring hosts
h1 h2
*** Starting controller
c0
*** Starting 1 switches
s1 ...
*** Waiting for switches to connect
s1
*** Ping: testing ping reachability
h1 -> h2
h2 -> h1
*** Results: 0% dropped (2/2 received)
*** Stopping 1 controllers
c0
*** Stopping 2 links
..
*** Stopping 1 switches
s1
*** Stopping 2 hosts
h1 h2
*** Done
completed in 0.476 seconds
```
## 3. Install Wireshark
Install ```Wireshark``` on the ubuntu machine.

See this [reference](https://www.cherryservers.com/blog/install-wireshark-ubuntu). Basically, you can use the ```apt install```.

You can run Wireshark by 
```bash
sudo wireshark &
```

If you want to enable non-superusers to capture packets (with the installation prompt window or using the ```udo dpkg-reconfigure wireshark-common``` command), add your user to the wireshark and/or sudo groups.

```bash
sudo usermod -aG wireshark "username" # optional
sudo usermod -aG sudo "username" # optional
```


Note: If you encounter an authentication error like ```authorization required, but no authorization protocol specified```, you can remove all entries in ```~/.Xauthority``` by making the file empty.

To start capturing, for example, capture ```s1-eth1```.


## 4. GUI
You can use GUI through ```Windows App``` (used to be called ```Microsoft remote desktop```) on macos or ```“Remote Desktop Connection``` on Windows.

See this [reference](https://documentation.ubuntu.com/multipass/latest/how-to-guides/customise-multipass/set-up-a-graphical-interface/)

# Initial Experiments [in class]
## Observe ARP and ICMP packets on a switch port
- Send a ping packet and observe the packet exchanged at a switch port (e.g., ```s1-eth1```)
```bash
mininet> h1 ping -c 1 h2
```

## Observe OpenFlow packets on the Loopback port (```lo```)
- Send a ping packet and observe the openflow messages (e.g., ```OFPT_PACKET_IN```) sent to a controller at ```lo```
```bash
mininet> h1 ping -c 1 h2
```

- ```OFPT_PACKET_IN``` is an OpenFlow message sent from an OpenFlow switch to its controller to transfer a packet that the switch has received but cannot forward based on its current rules. This typically happens when there is a "table-miss" — meaning no matching flow entry exists for the packet — and the switch is configured to forward the packet to the controller for a decision. The message contains **the packet data** and information about which port the packet arrived on. 
- The controller receives the packet and can decide how to handle it by sending a different message back to the switch, such as an ```OFPT_PACKET_OUT``` message to forward the packet to a specific port.