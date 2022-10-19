# Ships and Penguins

Ships and Penguins is a game based on particle systems developed by two students. We wrote our own physics engine from scratch within the scope this project. It includes simulations of fluid, cloth and rigid body particles interacting with each other. We developed this game with CUDA and C++ and used the rendering framework saiga.

https://user-images.githubusercontent.com/57756729/154817710-a8371afd-39f2-402e-852e-96d57c8ab172.mp4

# Game Description

The player navigates a boat equipped with a cannon through the sea using the arrow keys. The cannon is operated by the mouse. Using the space bar, the player can fire two different types of fish: A heavy and round puffer fish and a lighter and elongated swordfish. These can be exchanged with the number keys. After firing, the player must wait until the cannon has been reloaded.

Points are scored by hitting the sail of an enemy boat or the boat itself. In addition, extra points are awarded for hitting an enemy penguin.
If after three shots not a single point has been scored or the timer has expired, the game is over and the high score appears.

https://user-images.githubusercontent.com/57756729/196751071-6f894c06-afff-42b4-a7c2-53a2ebb7c800.mp4

# Simulating the Ocean

To enlarge the size of the ocean optically, we make use of trochoidal wave particles at the boundaries of the playable region. These particles are very cheap to simulate, as each follows a closed circle trajectory. This creates a regular wave pattern as seen in the first scene of the video.

We generate several of these regular trochoidal waves of various directions, wave lengths, and steepness and combine them. This results in nice irregular ocean waves. As only the surface of the ocean is of relevance to us, we apply only the uppermost two layers, which drastically reduces the number of particles.


https://user-images.githubusercontent.com/57756729/196764214-887d8a01-85a4-481c-9b25-7a438d227d4f.mp4

