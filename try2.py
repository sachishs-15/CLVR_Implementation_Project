class PPO_images(nn.Module):

    def __init__(self, env, encoder, args, state_dim = 64):

        super().__init__()

        self.env = env
        self.state_dim = state_dim
        self.action_dim = self.env.action_space.shape[0]

        self.mini_epochs = args.mini_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epochs = args.epochs
        self.clip = args.clip
        self.rollout_mem = args.rollout_mem
        self.log = args.log
        self.lr = args.lr
        self.freeze = args.freeze
        self.encoder_lr = args.encoder_lr
        self.batch_size = args.batch_size

        self.gamma = 0.99

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)            

        self.encoder = encoder
        self.encoder.to(self.device)

        ac_hidden = 64 # hidden layer size for actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, ac_hidden, self.cov_mat).to(self.device)
        self.critic = Critic(self.state_dim, ac_hidden).to(self.device)

        param = list(self.actor.parameters()) + list(self.critic.parameters()) 
        if not self.freeze:
            param = param + list(self.encoder.parameters()) # add encoder parameters to optimizer

        self.combined_optimizer = torch.optim.Adam((param), lr=self.lr)

    def rollout_data(self):

        rewards = []; actions = []; probs = []; values = []; states = []; encoded_states = []; discounted_rewards = []
        ep_rewards = []

        iters = 0

        while iters < self.rollout_mem:

            obs = self.env.reset()
            done = False
            rewards = []

            while not done:
                
                #set_trace()
                obs = np.array(obs)
                obs_encoded = self.encoder(torch.tensor(obs, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)).detach()
                obs_encoded = obs_encoded.squeeze()
            
                dist = self.actor(obs_encoded)
                action = dist.sample()
                prob = dist.log_prob(action).detach()
                
                state, reward, done, _ = self.env.step(action.cpu().detach())

                if done: 
                    break
                
                iters +=1 

                encoded_states.append(obs_encoded)
                states.append(obs)
                rewards.append(reward)
                actions.append(action)
                probs.append(prob)

                obs = state
            
            discounted_rewards = discounted_rewards + self.discounted_rewards(rewards)
            ep_rewards.append(sum(rewards))

            if iters % 5 == 0: 
                if self.log:
                    wandb.log({"mean rewards": np.mean(ep_rewards)})
                ep_rewards = []

        encoded_states = torch.stack(encoded_states)
        values = self.critic(encoded_states).detach()
        
        return torch.tensor(np.array(states), dtype=torch.float).unsqueeze(1).to(self.device), encoded_states, values, torch.stack(actions), torch.stack(probs), torch.tensor(discounted_rewards, dtype=torch.float).to(self.device)
    
    def discounted_rewards(self, rewards):

        discounted_rewards = []
        G = 0
        rewards.reverse()

        for i in range(len(rewards)):
            G = rewards[i] + self.gamma * G
            discounted_rewards.insert(0, G)
    
        return discounted_rewards
    
    def update_policy(self):
        
        #set_trace()
        states, enc_states, values, actions, probs, discounted_rewards = self.rollout_data()
    
        for data in range(self.mini_epochs):

            indices = np.random.choice(range(self.rollout_mem), self.batch_size)
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

            states_x = torch.index_select(states, 0, indices)
            enc_states_x = torch.index_select(enc_states, 0, indices)
            values_x = torch.index_select(values, 0, indices)
            actions_x = torch.index_select(actions, 0, indices)
            old_probs_x = torch.index_select(probs, 0, indices)
            discounted_rewards_x = torch.index_select(discounted_rewards, 0, indices)
            
            #if not self.freeze:
                #enc_states_x = self.encoder(states_x)
                #self.encoder_optimizer.zero_grad()

            dist = self.actor(enc_states_x)
            new_probs = dist.log_prob(actions_x)
            
            prob_ratio = new_probs.exp() / old_probs_x.exp()
        
            with torch.no_grad():
                advantage = discounted_rewards_x - values_x

            surr1 = prob_ratio * advantage
            surr2 = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage
            actor_loss = -(torch.min(surr1, surr2)).mean()

            x = self.critic(enc_states_x)
            x = x.squeeze()
            discounted_rewards_x = discounted_rewards_x.squeeze()
            critic_loss = nn.MSELoss()(x, discounted_rewards_x)

            if self.log:
                wandb.log({"actor loss": actor_loss.item()})
                wandb.log({"critic loss": critic_loss.item()})
            
            total_loss = actor_loss + 0.5 * critic_loss


            self.combined_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()) , 1)
            self.combined_optimizer.step()

            #if not self.freeze:
              #  self.encoder_optimizer.step()

    def train(self):

        for k in range(self.epochs):
            self.update_policy()
    