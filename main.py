import csv
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import random

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

csv_filename = "conversation_data.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Round", "Player", "Role", "Response", "evaluation"])

    # List of all players
    players = ["Claire", "Thomas", "Emma", "Lucas", "Laura", "Antoine", "Marie"]

    # Initialize roles with empty lists
    roles = {player: [] for player in players}

    # Function to assign roles for a round
    def assign_roles():
        # Assign roles for each player
        for player in players:
            roles[player] = []  # Reset roles for each player
        # Assign Loups-Garous roles
        for _ in range(2):
            player = random.choice(players)
            while len(roles[player]) > 0:  # Ensure player doesn't already have a role
                player = random.choice(players)
            roles[player].append("Loups-Garous")
        # Assign Voyante role
        player = random.choice(players)
        while len(roles[player]) > 0:  # Ensure player doesn't already have a role
            player = random.choice(players)
        roles[player].append("Voyante")
        # Assign Docteur role
        player = random.choice(players)
        while len(roles[player]) > 0:  # Ensure player doesn't already have a role
            player = random.choice(players)
        roles[player].append("Docteur")
        # Assign remaining players as Villageois
        for player in players:
            if len(roles[player]) == 0:
                roles[player].append("Villageois")

    print("Welcome to the Le Loup Garou game!\n")


    # Define evaluation functions for each role
    loup_garou_players = [p for p, r in roles.items() if "Loups-Garous" in r]
    voyante_players = [p for p, r in roles.items() if "Voyante" in r]
    villageois_players = [p for p, r in roles.items() if "Villageois" in r]
    def evaluate_loups_garous(response):
        if "innocent" in response.lower() or "simple villager" in response.lower() or "regret it" in response.lower() or "not the werewolf" in response.lower():
            return "False"
        elif "in my opinion" in response.lower():
            return "False"
        else:
            return "True"


    def evaluate_voyante(response):
        return "True"

    def evaluate_docteur(response):
        return "True"

    def evaluate_villageois(response):
        if "are we forgetting about" in response.lower() or "sudden interest in protecting the villagers" in response.lower() or "Looks like we've got a tie" in response.lower() or "you are wrong" in response.lower():
            return "True"
            # "the werewolf is" in response.lower() and "Loups-Garous" in roles[prompt_player]:
        elif "i suspect that" in response.lower():
            return "False"
        else:
            return "True"




    for round_num in range(1, 101):
        print(f"\nRound {round_num}:")
        assign_roles()  # Assign roles for the current round

        exchanges_count = 0
        while exchanges_count < 10:
            for player, player_roles in roles.items():
                role = player_roles[0]  # Take the first role for the player for this round
                print(f"{player} ({role}):")

                # Choose another player's name for the prompt
                prompt_players = [p for p in players if p != player]
                if prompt_players:
                    prompt_player = random.choice(prompt_players)
                else:
                    prompt_player = player

                # Modify the prompt based on the player's role and the chosen prompt player
                if role == "Loups-Garous":
                    loup_garou_players = [p for p, r in roles.items() if "Loups-Garous" in r]
                    voyante_players = [p for p, r in roles.items() if "Voyante" in r]
                    villageois_players = [p for p, r in roles.items() if "Villageois" in r]

                    if loup_garou_players:  # Ensure there are Loups-Garous assigned for this round
                        loup_garou_name = random.choice(loup_garou_players)
                        voyante_name = random.choice([p for p in prompt_players if p in voyante_players and p != loup_garou_name])
                        villageois_name = random.choice([p for p in prompt_players if p != loup_garou_name and p != voyante_name])

                        loup_prompt = ["I am innocent... I am a simple villager.", "You will regret it, I'm not the werewolf.", f"in my opinion {villageois_name} is the werewolf.", f"{voyante_name} is the voyante i think "]
                        prompt = random.choice(loup_prompt)
                        response = pipe(prompt, max_length=85, num_return_sequences=1, temperature=0.3)[0]['generated_text']
                        evaluation = evaluate_loups_garous(response)

                    else:
                        prompt = "Alright folks, it's time for another round of Loup Garou. Last night was intense, let's see who got turned."
                        response = pipe(prompt, max_length=85, num_return_sequences=1, temperature=0.3)[0]['generated_text']
                        evaluation = evaluate_loups_garous(response)

                elif role == "Voyante":
                    loup_garou_players = [p for p, r in roles.items() if "Loups-Garous" in r]
                    loup_garou_name = random.choice(loup_garou_players)
                    prompt = f"I think {loup_garou_name} is the werewolf."
                    response = pipe(prompt, max_length=85, num_return_sequences=1, temperature=0.3)[0]['generated_text']
                    evaluation = evaluate_voyante(response)
                elif role == "Docteur":
                    loup_garou_players = [p for p, r in roles.items() if "Loups-Garous" in r]
                    if loup_garou_players:  # Ensure there are Loups-Garous assigned for this round
                        loup_garou_name = random.choice(loup_garou_players)
                        if villageois_players:
                            docteur_prompt = [f"Well, I have my suspicions. {loup_garou_name} was awfully quiet during the accusations last night. I think it's a classic werewolf behavior if you ask me.", f"I decide to protect {random.choice(villageois_players)}."]
                            prompt = random.choice(docteur_prompt)
                            response = pipe(prompt, max_length=85, num_return_sequences=1, temperature=0.3)[0]['generated_text']
                            evaluation = evaluate_docteur(response)
                    else:
                        prompt = "i don't know what to say i'm confused"
                else:  # Villageois
                    loup_garou_players = [p for p, r in roles.items() if "Loups-Garous" in r]
                    voyante_players = [p for p, r in roles.items() if "Voyante" in r]
                    villageois_players = [p for p, r in roles.items() if "Villageois" in r]
                    if loup_garou_players:  # Ensure there are Loups-Garous assigned for this round
                        loup_garou_name = random.choice(loup_garou_players)
                        voyante_name = random.choice([p for p in prompt_players if p in voyante_players and p != loup_garou_name])
                        villageois_name = random.choice([p for p in prompt_players if p != loup_garou_name and p != voyante_name])
                        villageois_prompts = [
                            f"Hold on a second, are we forgetting about {loup_garou_name}? They've been playing the 'I'm just a simple villager' card a bit too much for my liking.",
                            f"I agree with {voyante_name}. {loup_garou_name}'s sudden interest in protecting the villagers seems a bit suspicious to me.",
                            f"The werewolf is {loup_garou_name}.",
                            f"i suspect that {villageois_name} is the werewolf.",
                            f"looks like we've got a tie between {villageois_name} and {loup_garou_name}. Loup Garou, reveal yourself!",
                        ]
                        prompt = random.choice(villageois_prompts)
                        response = pipe(prompt, max_length=85, num_return_sequences=1, temperature=0.3)[0]['generated_text']
                        evaluation = evaluate_villageois(response)

                    else:
                        prompt = "You are wrong! I am a villager... I am innocent... I am not the werewolf."
                        response = pipe(prompt, max_length=85, num_return_sequences=1, temperature=0.3)[0]['generated_text']
                        evaluation = evaluate_villageois(response)

                #response = pipe(prompt, max_length=85, num_return_sequences=1, temperature=0.3)[0]['generated_text']
                #evaluation = evaluate_loups_garous(response)

                # Extracting the response from the generated text
                # response = response[len(prompt):].strip()

                #if not response.endswith((".", "!", "?")):
                    #response += "."



                print(response)

                # Write data to CSV with semicolon delimiter
                csv_writer.writerow([round_num, player, role, response, evaluation])
                exchanges_count += 1
                if exchanges_count >= 10:
                    break

    print("\nThe game has ended.")
