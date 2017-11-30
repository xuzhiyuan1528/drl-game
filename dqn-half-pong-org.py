from Agent.dqn_half_pong_agent_org import DeepQHalfPongPlayer

if __name__ == '__main__':
    # Here, it loads the trained model and plays the game.
    # Please note that the model was trained with 40*40 grid at 8 frames per
    # second, so the game window is very small.
    player = DeepQHalfPongPlayer(
        checkpoint_path='./Res/deep_q_half_pong_networks_40x40_8',
        playback_mode=False,
        verbose_logging=False
    )

    # To train the model, uncomment the following code.
    # player = DeepQHalfPongPlayer()

    player.start()
