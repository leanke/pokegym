from pokegym import ram_map

class Events:
    gym_leader = 5
    gym_trainer = 3
    gym_task = 2
    trainer = 1
    hm = 5
    tm = 2
    task = 2
    pokemon = 10
    item = 5
    bill_capt = 5
    rival = 3
    quest = 5
    event = 1
    bad = -1
    

    def __init__(self, game, time):
        self.game = game
        self.time = time
        self._cached_rewards = None
        self.reward_calc = 0
        self.poketower = [142, 143, 144, 145, 146, 147, 148]
        self.pokehideout = [199, 200, 201, 202, 203, 135] # , 135
        self.silphco = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        self.rock_tunnel = [82, 232]
        self.ssanne = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
        self.mtmoon = [59, 60, 61]
        self.mansion = [214, 215, 216, 165]
        self.event_bits = {
            'found_rocket_hideout': (self.quest, 0xD77E, 1),
            'beat_rocket_hideout_giovanni': (self.gym_leader, 0xD81B, 7),
            'rescued_mr_fuji': (self.task, 0xD7E0, 7),
            'got_poke_flute': (self.quest, 0xD76C, 0),
            "beat_Silph_Co_Giovanni": (self.gym_leader, 0xD838, 7),
            'hm03': (self.hm, 0xD857, 0),
            'gave_gold_teeth': (self.quest, 0xD78E, 1),
            'got_bicycle': (self.quest, 0xD75F, 0),
            'route12_snorlax_beat': (self.pokemon, 0xD7D8, 7),
            'route16_snorlax_beat': (self.pokemon, 0xD7E0, 1),
            'brock': (self.gym_leader, 0xD755, 7),
            'misty': (self.gym_leader, 0xD75E, 7),
            'surge': (self.gym_leader, 0xD773, 7),
            'erika': (self.gym_leader, 0xD77C, 1),
            'koga': (self.gym_leader, 0xD792, 1),
            'sabrina': (self.gym_leader, 0xD7B3, 1),
            'blaine': (self.gym_leader, 0xD79A, 1),
            'giovanni': (self.gym_leader, 0xD751, 1),
        }
        self.silphco_bits = {
            "beat_Silph_Co_2F_trainer_0": (self.trainer, 0xD825, 2),
            "beat_Silph_Co_2F_trainer_1": (self.trainer, 0xD825, 3),
            "beat_Silph_Co_2F_trainer_2": (self.trainer, 0xD825, 4),
            "beat_Silph_Co_2F_trainer_3": (self.trainer, 0xD825, 5),
            "Silph_Co_2_Unlocked_Door1": (self.quest, 0xD826, 5),
            "Silph_Co_2_Unlocked_Door2": (self.quest, 0xD826, 6),
            "beat_Silph_Co_3F_trainer_0": (self.trainer, 0xD827, 2),
            "beat_Silph_Co_3F_trainer_1": (self.trainer, 0xD827, 3),
            "Silph_Co_3_Unlocked_Door1": (self.quest, 0xD828, 0),
            "Silph_Co_3_Unlocked_Door2": (self.quest, 0xD828, 1),
            "beat_Silph_Co_4F_trainer_0": (self.trainer, 0xD829, 2),
            "beat_Silph_Co_4F_trainer_1": (self.trainer, 0xD829, 3),
            "beat_Silph_Co_4F_trainer_2": (self.trainer, 0xD829, 4),
            "Silph_Co_4_Unlocked_Door1": (self.quest, 0xD82A, 0),
            "Silph_Co_4_Unlocked_Door2": (self.quest, 0xD82A, 1),
            "beat_Silph_Co_5F_trainer_0": (self.trainer, 0xD82B, 2),
            "beat_Silph_Co_5F_trainer_1": (self.trainer, 0xD82B, 3),
            "beat_Silph_Co_5F_trainer_2": (self.trainer, 0xD82B, 4),
            "beat_Silph_Co_5F_trainer_3": (self.trainer, 0xD82B, 5),
            "Silph_Co_5_Unlocked_Door1": (self.quest, 0xD82C, 0),
            "Silph_Co_5_Unlocked_Door2": (self.quest, 0xD82C, 1),
            "Silph_Co_5_Unlocked_Door3": (self.quest, 0xD82C, 2),
            "beat_Silph_Co_6F_trainer_0": (self.trainer, 0xD82D, 6),
            "beat_Silph_Co_6F_trainer_1": (self.trainer, 0xD82D, 7),
            "beat_Silph_Co_6F_trainer_2": (self.trainer, 0xD82E, 0),
            "Silph_Co_6_Unlocked_Door": (self.quest, 0xD82E, 7),
            "beat_Silph_Co_7F_trainer_0": (self.trainer, 0xD82F, 5),
            "beat_Silph_Co_7F_trainer_1": (self.trainer, 0xD82F, 6),
            "beat_Silph_Co_7F_trainer_2": (self.trainer, 0xD82F, 7),
            "beat_Silph_Co_7F_trainer_3": (self.trainer, 0xD830, 0),
            "Silph_Co_7_Unlocked_Door1": (self.quest, 0xD830, 4),
            "Silph_Co_7_Unlocked_Door2": (self.quest, 0xD830, 5),
            "Silph_Co_7_Unlocked_Door3": (self.quest, 0xD830, 6),
            "beat_Silph_Co_8F_trainer_0": (self.trainer, 0xD831, 2),
            "beat_Silph_Co_8F_trainer_1": (self.trainer, 0xD831, 3),
            "beat_Silph_Co_8F_trainer_2": (self.trainer, 0xD831, 4),
            "Silph_Co_8_Unlocked_Door": (self.quest, 0xD832, 0),
            "beat_Silph_Co_9F_trainer_0": (self.trainer, 0xD833, 2),
            "beat_Silph_Co_9F_trainer_1": (self.trainer, 0xD833, 3),
            "beat_Silph_Co_9F_trainer_2": (self.trainer, 0xD833, 4),
            "Silph_Co_9_Unlocked_Door1": (self.quest, 0xD834, 0),
            "Silph_Co_9_Unlocked_Door2": (self.quest, 0xD834, 1),
            "Silph_Co_9_Unlocked_Door3": (self.quest, 0xD834, 2),
            "Silph_Co_9_Unlocked_Door4": (self.quest, 0xD834, 3),
            "beat_Silph_Co_10F_trainer_0": (self.trainer, 0xD835, 1),
            "beat_Silph_Co_10F_trainer_1": (self.trainer, 0xD835, 2),
            "Silph_Co_10_Unlocked_Door": (self.quest, 0xD836, 0),
            "beat_Silph_Co_11F_trainer_0": (self.trainer, 0xD837, 4),
            "beat_Silph_Co_11F_trainer_1": (self.trainer, 0xD837, 5),
            "Silph_Co_11_Unlocked_Door": (self.quest, 0xD838, 0),
            "Got_Master_Ball": (self.item, 0xD838, 5),
            "beat_Silph_Co_Giovanni": (self.gym_leader, 0xD838, 7),
            "Silph_Co_Receptionist_At_Desk": (self.task, 0xD7B9, 7),
        }
        self.rock_tunnel_bits = {
            'beat_Rock_Tunnel_1_trainer_0': (self.trainer, 0xD7D2, 1),
            'beat_Rock_Tunnel_1_trainer_1': (self.trainer, 0xD7D2, 2),
            'beat_Rock_Tunnel_1_trainer_2': (self.trainer, 0xD7D2, 3),
            'beat_Rock_Tunnel_1_trainer_3': (self.trainer, 0xD7D2, 4),
            'beat_Rock_Tunnel_1_trainer_4': (self.trainer, 0xD7D2, 5),
            'beat_Rock_Tunnel_1_trainer_5': (self.trainer, 0xD7D2, 6),
            'beat_Rock_Tunnel_1_trainer_6': (self.trainer, 0xD7D2, 7),
            'beat_Rock_Tunnel_2_trainer_0': (self.trainer, 0xD87D, 1),
            'beat_Rock_Tunnel_2_trainer_1': (self.trainer, 0xD87D, 2),
            'beat_Rock_Tunnel_2_trainer_2': (self.trainer, 0xD87D, 3),
            'beat_Rock_Tunnel_2_trainer_3': (self.trainer, 0xD87D, 4),
            'beat_Rock_Tunnel_2_trainer_4': (self.trainer, 0xD87D, 5),
            'beat_Rock_Tunnel_2_trainer_5': (self.trainer, 0xD87D, 6),
            'beat_Rock_Tunnel_2_trainer_6': (self.trainer, 0xD87D, 7),
            'beat_Rock_Tunnel_2_trainer_7': (self.trainer, 0xD87E, 0),
        }
        self.ssanne_bits = {
            'beat_Ss_Anne_5_trainer_0': (self.trainer, 0xD7FF, 4),
            'beat_Ss_Anne_5_trainer_1': (self.trainer, 0xD7FF, 5),
            'Rubbed_Captains_Back': (self.bill_capt, 0xD803, 1),
            'Ss_Anne_Left': (self.bill_capt, 0xD803, 2),
            'Walked_Past_Guard_After_Ss_Anne_Left': (self.bill_capt, 0xD803, 3),
            'Started_Walking_Out_Of_Dock': (self.bill_capt, 0xD803, 4),
            'Walked_Out_Of_Dock': (self.bill_capt, 0xD803, 5),
            'beat_Ss_Anne_8_trainer_0': (self.trainer, 0xD805, 1),
            'beat_Ss_Anne_8_trainer_1': (self.trainer, 0xD805, 2),
            'beat_Ss_Anne_8_trainer_2': (self.trainer, 0xD805, 3),
            'beat_Ss_Anne_8_trainer_3': (self.trainer, 0xD805, 4),
            'beat_Ss_Anne_9_trainer_0': (self.trainer, 0xD807, 1),
            'beat_Ss_Anne_9_trainer_1': (self.trainer, 0xD807, 2),
            'beat_Ss_Anne_9_trainer_2': (self.trainer, 0xD807, 3),
            'beat_Ss_Anne_9_trainer_3': (self.trainer, 0xD807, 4),
            'beat_Ss_Anne_10_trainer_0': (self.trainer, 0xD809, 1),
            'beat_Ss_Anne_10_trainer_1': (self.trainer, 0xD809, 2),
            'beat_Ss_Anne_10_trainer_2': (self.trainer, 0xD809, 3),
            'beat_Ss_Anne_10_trainer_3': (self.trainer, 0xD809, 4),
            'beat_Ss_Anne_10_trainer_4': (self.trainer, 0xD809, 5),
            'beat_Ss_Anne_10_trainer_5': (self.trainer, 0xD809, 6),
        }
        self.mtmoon_bits = {
            'beat_Mt_Moon_1_trainer_1': (self.trainer, 0xD7F5, 1),
            'beat_Mt_Moon_1_trainer_2': (self.trainer, 0xD7F5, 2),
            'beat_Mt_Moon_1_trainer_3': (self.trainer, 0xD7F5, 3),
            'beat_Mt_Moon_1_trainer_4': (self.trainer, 0xD7F5, 4),
            'beat_Mt_Moon_1_trainer_5': (self.trainer, 0xD7F5, 5),
            'beat_Mt_Moon_1_trainer_6': (self.trainer, 0xD7F5, 6),
            'beat_Mt_Moon_1_trainer_7': (self.trainer, 0xD7F5, 7),
            'beat_Mt_Moon_Super_Nerd': (self.trainer, 0xD7F6, 1),
            'beat_Mt_Moon_3_trainer_0': (self.trainer, 0xD7F6, 2),
            'beat_Mt_Moon_3_trainer_1': (self.trainer, 0xD7F6, 3),
            'beat_Mt_Moon_3_trainer_2': (self.trainer, 0xD7F6, 4),
            'beat_Mt_Moon_3_trainer_3': (self.trainer, 0xD7F6, 5),
            'Got_Dome_Fossil': (self.task, 0xD7F6, 6),
            'Got_Helix_Fossil': (self.task, 0xD7F6, 7),
        }
        self.route3_bits = {
            'route3_0': (self.trainer, 0xD7C3, 2),
            'route3_1': (self.trainer, 0xD7C3, 3),
            'route3_2': (self.trainer, 0xD7C3, 4),
            'route3_3': (self.trainer, 0xD7C3, 5),
            'route3_4': (self.trainer, 0xD7C3, 6),
            'route3_5': (self.trainer, 0xD7C3, 7),
            'route3_6': (self.trainer, 0xD7C4, 0),
            'route3_7': (self.trainer, 0xD7C4, 1),
        }
        self.route4_bits = {
            'route4_0': (self.trainer, 0xD7C5, 2),
        }
        self.route24_bits = {
            'route24_rocket': (self.trainer, 0xD7EF, 1),
            'route24_0': (self.trainer, 0xD7EF, 2),
            'route24_1': (self.trainer, 0xD7EF, 3),
            'route24_2': (self.trainer, 0xD7EF, 4),
            'route24_3': (self.trainer, 0xD7EF, 5),
            'route24_4': (self.trainer, 0xD7EF, 6),
            'route24_5': (self.trainer, 0xD7EF, 7),
        }
        self.route25_bits = {
            'route25_0': (self.trainer, 0xD7F1, 1),
            'route25_1': (self.trainer, 0xD7F1, 2),
            'route25_2': (self.trainer, 0xD7F1, 3),
            'route25_3': (self.trainer, 0xD7F1, 4),
            'route25_4': (self.trainer, 0xD7F1, 5),
            'route25_5': (self.trainer, 0xD7F1, 6),
            'route25_6': (self.trainer, 0xD7F1, 7),
            'route25_7': (self.trainer, 0xD7F2, 0),
            'route25_8': (self.trainer, 0xD7F2, 1),
        }
        self.route9_bits = {
            'route9_0': (self.trainer, 0xD7CF, 1),
            'route9_1': (self.trainer, 0xD7CF, 2),
            'route9_2': (self.trainer, 0xD7CF, 3),
            'route9_3': (self.trainer, 0xD7CF, 4),
            'route9_4': (self.trainer, 0xD7CF, 5),
            'route9_5': (self.trainer, 0xD7CF, 6),
            'route9_6': (self.trainer, 0xD7CF, 7),
            'route9_7': (self.trainer, 0xD7D0, 0),
            'route9_8': (self.trainer, 0xD7D0, 1),
        }
        self.route6_bits = {
            'route6_0': (self.trainer, 0xD7C9, 1),
            'route6_1': (self.trainer, 0xD7C9, 2),
            'route6_2': (self.trainer, 0xD7C9, 3),
            'route6_3': (self.trainer, 0xD7C9, 4),
            'route6_4': (self.trainer, 0xD7C9, 5),
            'route6_5': (self.trainer, 0xD7C9, 6),
        }
        self.route11_bits = {
            'route11_0': (self.trainer, 0xD7D5, 1),
            'route11_1': (self.trainer, 0xD7D5, 2),
            'route11_2': (self.trainer, 0xD7D5, 3),
            'route11_3': (self.trainer, 0xD7D5, 4),
            'route11_4': (self.trainer, 0xD7D5, 5),
            'route11_5': (self.trainer, 0xD7D5, 6),
            'route11_6': (self.trainer, 0xD7D5, 7),
            'route11_7': (self.trainer, 0xD7D6, 0),
            'route11_8': (self.trainer, 0xD7D6, 1),
            'route11_9': (self.trainer, 0xD7D6, 2),
        }
        self.route8_bits = {
            'route8_0': (self.trainer, 0xD7CD, 1),
            'route8_1': (self.trainer, 0xD7CD, 2),
            'route8_2': (self.trainer, 0xD7CD, 3),
            'route8_3': (self.trainer, 0xD7CD, 4),
            'route8_4': (self.trainer, 0xD7CD, 5),
            'route8_5': (self.trainer, 0xD7CD, 6),
            'route8_6': (self.trainer, 0xD7CD, 7),
            'route8_7': (self.trainer, 0xD7CE, 0),
            'route8_8': (self.trainer, 0xD7CE, 1),
        }
        self.route10_bits = {
            'route10_0': (self.trainer, 0xD7D1, 1),
            'route10_1': (self.trainer, 0xD7D1, 2),
            'route10_2': (self.trainer, 0xD7D1, 3),
            'route10_3': (self.trainer, 0xD7D1, 4),
            'route10_4': (self.trainer, 0xD7D1, 5),
            'route10_5': (self.trainer, 0xD7D1, 6),
        }
        self.route12_bits = {
            'route12_0': (self.trainer, 0xD7D7, 2),
            'route12_1': (self.trainer, 0xD7D7, 3),
            'route12_2': (self.trainer, 0xD7D7, 4),
            'route12_3': (self.trainer, 0xD7D7, 5),
            'route12_4': (self.trainer, 0xD7D7, 6),
            'route12_5': (self.trainer, 0xD7D7, 7),
            'route12_6': (self.trainer, 0xD7D8, 0),
            'route12_snorlax_fight': (self.pokemon, 0xD7D8, 6),
            'route12_snorlax_beat': (self.pokemon, 0xD7D8, 7),
        }
        self.route16_bits = {
            'route16_snorlax_fight': (self.pokemon, 0xD7E0, 0),
            'route16_snorlax_beat': (self.pokemon, 0xD7E0, 1),
            'route16_0': (self.trainer, 0xD7DF, 1),
            'route16_1': (self.trainer, 0xD7DF, 2),
            'route16_2': (self.trainer, 0xD7DF, 3),
            'route16_3': (self.trainer, 0xD7DF, 4),
            'route16_4': (self.trainer, 0xD7DF, 5),
            'route16_5': (self.trainer, 0xD7DF, 6),
        }
        self.route17_bits = {
            'route17_0': (self.trainer, 0xD7E1, 1),
            'route17_1': (self.trainer, 0xD7E1, 2),
            'route17_2': (self.trainer, 0xD7E1, 3),
            'route17_3': (self.trainer, 0xD7E1, 4),
            'route17_4': (self.trainer, 0xD7E1, 5),
            'route17_5': (self.trainer, 0xD7E1, 6),
            'route17_6': (self.trainer, 0xD7E1, 7),
            'route17_7': (self.trainer, 0xD7E2, 0),
            'route17_8': (self.trainer, 0xD7E2, 1),
            'route17_9': (self.trainer, 0xD7E2, 2),
        }
        self.route13_bits = {
            'route13_0': (self.trainer, 0xD7D9, 1),
            'route13_1': (self.trainer, 0xD7D9, 2),
            'route13_2': (self.trainer, 0xD7D9, 3),
            'route13_3': (self.trainer, 0xD7D9, 4),
            'route13_4': (self.trainer, 0xD7D9, 5),
            'route13_5': (self.trainer, 0xD7D9, 6),
            'route13_6': (self.trainer, 0xD7D9, 7),
            'route13_7': (self.trainer, 0xD7DA, 0),
            'route13_8': (self.trainer, 0xD7DA, 1),
            'route13_9': (self.trainer, 0xD7DA, 2),
        }
        self.route14_bits = {
            'route14_0': (self.trainer, 0xD7DB, 1),
            'route14_1': (self.trainer, 0xD7DB, 2),
            'route14_2': (self.trainer, 0xD7DB, 3),
            'route14_3': (self.trainer, 0xD7DB, 4),
            'route14_4': (self.trainer, 0xD7DB, 5),
            'route14_5': (self.trainer, 0xD7DB, 6),
            'route14_6': (self.trainer, 0xD7DB, 7),
            'route14_7': (self.trainer, 0xD7DC, 0),
            'route14_8': (self.trainer, 0xD7DC, 1),
            'route14_9': (self.trainer, 0xD7DC, 2),
        }
        self.route15_bits = {
            'route15_0': (self.trainer, 0xD7DD, 1),
            'route15_1': (self.trainer, 0xD7DD, 2),
            'route15_2': (self.trainer, 0xD7DD, 3),
            'route15_3': (self.trainer, 0xD7DD, 4),
            'route15_4': (self.trainer, 0xD7DD, 5),
            'route15_5': (self.trainer, 0xD7DD, 6),
            'route15_6': (self.trainer, 0xD7DD, 7),
            'route15_7': (self.trainer, 0xD7DE, 0),
            'route15_8': (self.trainer, 0xD7DE, 1),
            'route15_9': (self.trainer, 0xD7DE, 2),
        }
        self.route18_bits = {
            'route18_0': (self.trainer, 0xD7E3, 1),
            'route18_1': (self.trainer, 0xD7E3, 2),
            'route18_2': (self.trainer, 0xD7E3, 3),
        }
        self.route19_bits = {
            'route19_0': (self.trainer, 0xD7E5, 1),
            'route19_1': (self.trainer, 0xD7E5, 2),
            'route19_2': (self.trainer, 0xD7E5, 3),
            'route19_3': (self.trainer, 0xD7E5, 4),
            'route19_4': (self.trainer, 0xD7E5, 5),
            'route19_5': (self.trainer, 0xD7E5, 6),
            'route19_6': (self.trainer, 0xD7E5, 7),
            'route19_7': (self.trainer, 0xD7E6, 0),
            'route19_8': (self.trainer, 0xD7E6, 1),
            'route19_9': (self.trainer, 0xD7E6, 2),
        }
        self.route20_bits = {
            'route20_0': (self.trainer, 0xD7E7, 1),
            'route20_1': (self.trainer, 0xD7E7, 2),
            'route20_2': (self.trainer, 0xD7E7, 3),
            'route20_3': (self.trainer, 0xD7E7, 4),
            'route20_4': (self.trainer, 0xD7E7, 5),
            'route20_5': (self.trainer, 0xD7E7, 6),
            'route20_6': (self.trainer, 0xD7E7, 7),
            'route20_7': (self.trainer, 0xD7E8, 0),
            'route20_8': (self.trainer, 0xD7E8, 1),
            'route20_9': (self.trainer, 0xD7E8, 2),
        }
        self.route21_bits = {
            'route21_0': (self.trainer, 0xD7E9, 1),
            'route21_1': (self.trainer, 0xD7E9, 2),
            'route21_2': (self.trainer, 0xD7E9, 3),
            'route21_3': (self.trainer, 0xD7E9, 4),
            'route21_4': (self.trainer, 0xD7E9, 5),
            'route21_5': (self.trainer, 0xD7E9, 6),
            'route21_6': (self.trainer, 0xD7E9, 7),
            'route21_7': (self.trainer, 0xD7EA, 0),
            'route21_8': (self.trainer, 0xD7EA, 1),
        }
        self.bill_bits = {
            'met_bill': (self.bill_capt, 0xD7F1, 0),
            'used_cell_separator_on_bill': (self.bill_capt, 0xD7F2, 3),
            'got_ss_ticket': (self.bill_capt, 0xD7F2, 4),
            'met_bill_2': (self.bill_capt, 0xD7F2, 5),
            'bill_said_use_cell_separator': (self.bill_capt, 0xD7F2, 6),
            'left_bills_house_after_helping': (self.bill_capt, 0xD7F2, 7),
        }
        self.misc_bits = {
            'hm01': (self.hm, 0xD803, 0),
            'hm02': (self.hm, 0xD7E0, 6),
            'hm03': (self.hm, 0xD857, 0),
            'hm04': (self.hm, 0xD78E, 0),
            'hm05': (self.hm, 0xD7C2, 0),
            'tm34': (self.tm, 0xD755, 6),
            'tm11': (self.tm, 0xD75E, 6),
            'tm41': (self.tm, 0xD777, 0),
            'tm13': (self.tm, 0xD778, 4),
            'tm48': (self.tm, 0xD778, 5),
            'tm49': (self.tm, 0xD778, 6),
            'tm18': (self.tm, 0xD778, 7),
            'tm21': (self.tm, 0xD77C, 0),
            'tm06': (self.tm, 0xD792, 0),
            'tm24': (self.tm, 0xD773, 6),
            'tm29': (self.tm, 0xD7BD, 0),
            'tm31': (self.tm, 0xD7AF, 0),
            'tm35': (self.tm, 0xD7A1, 7),
            'tm36': (self.tm, 0xD826, 7),
            'tm38': (self.tm, 0xD79A, 0),
            'tm27': (self.tm, 0xD751, 0),
            'tm42': (self.tm, 0xD74C, 1),
            'tm46': (self.tm, 0xD7B3, 0),
            'tm39': (self.tm, 0xD7D7, 0),
            'gave_fossil_to_lab': (self.task, 0xD7A3, 0),
            'lab_still_reviving_fossil': (self.task, 0xD7A3, 1),
            'lab_handing_over_fossil_mon': (self.task, 0xD7A3, 2),
            'gave_gold_teeth': (self.quest, 0xD78E, 1),
            'bought_magikarp': (self.task, 0xD7C6, 7),
            'hall_of_fame_dex_rating': (self.task, 0xD747, 3),
            'daisy_walking': (self.task, 0xD74A, 2),
            'got_old_amber': (self.task, 0xD754, 1),
            'got_bike_voucher': (self.task, 0xD771, 1),
            'got_10_coins': (self.task, 0xD77E, 2),
            'got_20_coins_1': (self.task, 0xD77E, 3),
            'got_20_coins_2': (self.task, 0xD77E, 4),
            'got_coin_case': (self.task, 0xD783, 0),
            'got_potion_sample': (self.task, 0xD7BF, 0),
            'got_itemfinder': (self.task, 0xD7D6, 7),
            'got_exp_all': (self.task, 0xD7DD, 0),
            'rescued_mr_fuji': (self.task, 0xD7E0, 7),
            'beat_mewtwo': (self.task, 0xD85F, 1),
            'rescued_mr_fuji_2': (self.task, 0xD769, 7),
            'oak_appeared_in_pallet': (self.task, 0xD74B, 7),
            'followed_oak_into_lab': (self.task, 0xD747, 0),
            'oak_asked_to_choose_mon': (self.task, 0xD74B, 1),
            'got_starter': (self.task, 0xD74B, 2),
            'followed_oak_into_lab_2': (self.task, 0xD74B, 0),
            'got_pokedex': (self.quest, 0xD74B, 5),
            'got_oaks_parcel': (self.quest, 0xD74E, 1),
            'pallet_after_getting_pokeballs': (self.quest, 0xD747, 6),
            'oak_got_parcel': (self.quest, 0xD74E, 0),
            'got_pokeballs_from_oak': (self.task, 0xD74B, 4),
            'pallet_after_getting_pokeballs_2': (self.task, 0xD74B, 6),
            'got_town_map': (self.task, 0xD74A, 0),
            'entered_blues_house': (self.task, 0xD74A, 1),
            'got_nugget': (self.task, 0xD7EF, 0),
            'nugget_reward_available': (self.task, 0xD7F0, 1),
            'beat_cerulean_rocket_thief': (self.trainer, 0xD75B, 7),
            'got_bicycle': (self.quest, 0xD75F, 0),
            'seel_fan_boast': (self.task, 0xD771, 6),
            'pikachu_fan_boast': (self.task, 0xD771, 7),
            'got_poke_flute': (self.quest, 0xD76C, 0),
            'rival_one': (self.rival, 0xD74B, 3),
            'rival_two': (self.rival, 0xD7EB, 0),
            'rival_three': (self.rival, 0xD7EB, 1),
            'rival_four': (self.rival, 0xD7EB, 5),
            'rival_five': (self.rival, 0xD7EB, 6),
            'rival_six': (self.rival, 0xD75A, 0),
            'rival_seven': (self.rival, 0xD764, 6),
            'rival_eight': (self.rival, 0xD764, 7),
            'rival_nine': (self.rival, 0xD7EB, 7),
            'beat_Silph_Co_rival': (self.rival, 0xD82F, 0),
            'viridian_gym_door_unlocked': (self.gym_task, 0xD74C, 0),
        }
        self.forest_bits = {
            'beat_viridian_forest_trainer_0': (self.trainer, 0xD7F3, 2),
            'beat_viridian_forest_trainer_1': (self.trainer, 0xD7F3, 3),
            'beat_viridian_forest_trainer_2': (self.trainer, 0xD7F3, 4),
        }
        self.mansion_bits = {
            'beat_mansion_2_trainer_0': (self.trainer, 0xD847, 1),
            'beat_mansion_3_trainer_0': (self.trainer, 0xD849, 1),
            'beat_mansion_3_trainer_1': (self.trainer, 0xD849, 2),
            'beat_mansion_4_trainer_0': (self.trainer, 0xD84B, 1),
            'beat_mansion_4_trainer_1': (self.trainer, 0xD84B, 2),
            'mansion_switch_on': (self.quest, 0xD796, 0),
            'beat_mansion_1_trainer_0': (self.trainer, 0xD798, 1),
        }
        self.dojo_bits = {
            'defeated_fighting_dojo': (self.task, 0xD7B1, 0),
            'beat_karate_master': (self.gym_leader, 0xD7B1, 1),
            'beat_dojo_trainer_0': (self.trainer, 0xD7B1, 2),
            'beat_dojo_trainer_1': (self.trainer, 0xD7B1, 3),
            'beat_dojo_trainer_2': (self.trainer, 0xD7B1, 4),
            'beat_dojo_trainer_3': (self.trainer, 0xD7B1, 5),
            'got_hitmonlee': (self.pokemon, 0xD7B1, 6),
            'got_hitmonchan': (self.pokemon, 0xD7B1, 7),
        }
        self.hideout_bits = {
            'beat_rocket_hideout_1_trainer_0': (self.gym_trainer, 0xD815, 1),
            'beat_rocket_hideout_1_trainer_1': (self.gym_trainer, 0xD815, 2),
            'beat_rocket_hideout_1_trainer_2': (self.gym_trainer, 0xD815, 3),
            'beat_rocket_hideout_1_trainer_3': (self.gym_trainer, 0xD815, 4),
            'beat_rocket_hideout_1_trainer_4': (self.gym_trainer, 0xD815, 5),
            'beat_rocket_hideout_2_trainer_0': (self.gym_trainer, 0xD817, 1),
            'beat_rocket_hideout_3_trainer_0': (self.gym_trainer, 0xD819, 1),
            'beat_rocket_hideout_3_trainer_1': (self.gym_trainer, 0xD819, 2),
            'beat_rocket_hideout_4_trainer_0': (self.gym_trainer, 0xD81B, 2),
            'beat_rocket_hideout_4_trainer_1': (self.gym_trainer, 0xD81B, 3),
            'beat_rocket_hideout_4_trainer_2': (self.gym_trainer, 0xD81B, 4),
            'rocket_hideout_4_door_unlocked': (self.quest, 0xD81B, 5),
            'rocket_dropped_lift_key': (self.quest, 0xD81B, 6),
            'beat_rocket_hideout_giovanni': (self.gym_leader, 0xD81B, 7),
            'found_rocket_hideout': (self.quest, 0xD77E, 1),
        }
        self.tower_bits = {
            'beat_pokemontower_3_trainer_0': (self.trainer, 0xD765, 1),
            'beat_pokemontower_3_trainer_1': (self.trainer, 0xD765, 2),
            'beat_pokemontower_3_trainer_2': (self.trainer, 0xD765, 3),
            'beat_pokemontower_4_trainer_0': (self.trainer, 0xD766, 1),
            'beat_pokemontower_4_trainer_1': (self.trainer, 0xD766, 2),
            'beat_pokemontower_4_trainer_2': (self.trainer, 0xD766, 3),
            'beat_pokemontower_5_trainer_0': (self.trainer, 0xD767, 2),
            'beat_pokemontower_5_trainer_1': (self.trainer, 0xD767, 3),
            'beat_pokemontower_5_trainer_2': (self.trainer, 0xD767, 4),
            'beat_pokemontower_5_trainer_3': (self.trainer, 0xD767, 5),
            'beat_pokemontower_6_trainer_0': (self.trainer, 0xD768, 1),
            'beat_pokemontower_6_trainer_1': (self.trainer, 0xD768, 2),
            'beat_pokemontower_6_trainer_2': (self.trainer, 0xD768, 3),
            'beat_ghost_marowak': (self.quest, 0xD768, 7),
            'beat_pokemontower_7_trainer_0': (self.trainer, 0xD769, 1),
            'beat_pokemontower_7_trainer_1': (self.trainer, 0xD769, 2),
            'beat_pokemontower_7_trainer_2': (self.trainer, 0xD769, 3),
        }
        self.brock_bits = {
            'brock': (self.gym_leader, 0xD755, 7),
            'beat_pewter_gym_trainer_0': (self.gym_trainer, 0xD755, 2),
        }
        self.misty_bits = {
            'misty': (self.gym_leader, 0xD75E, 7),
            'beat_cerulean_gym_trainer_0': (self.gym_trainer, 0xD75E, 2),
            'beat_cerulean_gym_trainer_1': (self.gym_trainer, 0xD75E, 3),
        }
        self.surge_bits = {
            'surge': (self.gym_leader, 0xD773, 7),
            'vermilion_gym_lock_one': (self.gym_task, 0xD773, 1),
            'vermilion_gym_lock_two': (self.gym_task, 0xD773, 0),
            'beat_vermilion_gym_trainer_0': (self.gym_trainer, 0xD773, 2),
            'beat_vermilion_gym_trainer_1': (self.gym_trainer, 0xD773, 3),
            'beat_vermilion_gym_trainer_2': (self.gym_trainer, 0xD773, 4),
        }
        self.erika_bits = {
            'erika': (self.gym_leader, 0xD77C, 1),
            'beat_celadon_gym_trainer_0': (self.gym_trainer, 0xD77C, 2),
            'beat_celadon_gym_trainer_1': (self.gym_trainer, 0xD77C, 3),
            'beat_celadon_gym_trainer_2': (self.gym_trainer, 0xD77C, 4),
            'beat_celadon_gym_trainer_3': (self.gym_trainer, 0xD77C, 5),
            'beat_celadon_gym_trainer_4': (self.gym_trainer, 0xD77C, 6),
            'beat_celadon_gym_trainer_5': (self.gym_trainer, 0xD77C, 7),
            'beat_celadon_gym_trainer_6': (self.gym_trainer, 0xD77D, 0),
        }
        self.koga_bits = {
            'koga': (self.gym_leader, 0xD792, 1),
            'beat_fuchsia_gym_trainer_0': (self.gym_trainer, 0xD792, 2),
            'beat_fuchsia_gym_trainer_1': (self.gym_trainer, 0xD792, 3),
            'beat_fuchsia_gym_trainer_2': (self.gym_trainer, 0xD792, 4),
            'beat_fuchsia_gym_trainer_3': (self.gym_trainer, 0xD792, 5),
            'beat_fuchsia_gym_trainer_4': (self.gym_trainer, 0xD792, 6),
            'beat_fuchsia_gym_trainer_5': (self.gym_trainer, 0xD792, 7),
        }
        self.sabrina_bits = {
            'sabrina': (self.gym_leader, 0xD7B3, 1),
            'beat_saffron_gym_trainer_0': (self.gym_trainer, 0xD7B3, 2),
            'beat_saffron_gym_trainer_1': (self.gym_trainer, 0xD7B3, 3),
            'beat_saffron_gym_trainer_2': (self.gym_trainer, 0xD7B3, 4),
            'beat_saffron_gym_trainer_3': (self.gym_trainer, 0xD7B3, 5),
            'beat_saffron_gym_trainer_4': (self.gym_trainer, 0xD7B3, 6),
            'beat_saffron_gym_trainer_5': (self.gym_trainer, 0xD7B3, 7),
            'beat_saffron_gym_trainer_6': (self.gym_trainer, 0xD7B4, 0),
        }
        self.blaine_bits = {
            'blaine': (self.gym_leader, 0xD79A, 1),
            'beat_cinnabar_gym_trainer_0': (self.gym_trainer, 0xD79A, 2),
            'beat_cinnabar_gym_trainer_1': (self.gym_trainer, 0xD79A, 3),
            'beat_cinnabar_gym_trainer_2': (self.gym_trainer, 0xD79A, 4),
            'beat_cinnabar_gym_trainer_3': (self.gym_trainer, 0xD79A, 5),
            'beat_cinnabar_gym_trainer_4': (self.gym_trainer, 0xD79A, 6),
            'beat_cinnabar_gym_trainer_5': (self.gym_trainer, 0xD79A, 7),
            'beat_cinnabar_gym_trainer_6': (self.gym_trainer, 0xD79B, 0),
        }
        self.giovanni_bits = {
            'giovanni': (self.gym_leader, 0xD751, 1),
            'beat_viridian_gym_trainer_0': (self.gym_trainer, 0xD751, 2),
            'beat_viridian_gym_trainer_1': (self.gym_trainer, 0xD751, 3),
            'beat_viridian_gym_trainer_2': (self.gym_trainer, 0xD751, 4),
            'beat_viridian_gym_trainer_3': (self.gym_trainer, 0xD751, 5),
            'beat_viridian_gym_trainer_4': (self.gym_trainer, 0xD751, 6),
            'beat_viridian_gym_trainer_5': (self.gym_trainer, 0xD751, 7),
            'beat_viridian_gym_trainer_6': (self.gym_trainer, 0xD752, 0),
            'beat_viridian_gym_trainer_7': (self.gym_trainer, 0xD752, 1),
        }

    def read_bit(self, address, bit) -> bool:
        return bin(256 + self.game.get_memory_value(address))[-bit - 1] == "1"

    def calculate_reward(self, base_value, address, bit):
        return base_value * int(self.read_bit(address, bit))
    
    def bit_check(self, key):
        value = self.event_bits[key]
        _, address, bit = value
        return int(self.read_bit(address, bit))
    
    def sum_event_rewards(self):
        rewards = self.event_rewards()
        return sum(rewards.values())
    
    def pick_dict(self):
        _, _, map_n = ram_map.position(self.game)
        if map_n == 14:
            return self.route3_bits
        elif map_n == 15:
            return self.route4_bits
        elif map_n == 17:
            return self.route6_bits
        elif map_n == 19:
            return self.route8_bits
        elif map_n == 20:
            return self.route9_bits
        elif map_n == 21:
            return self.route10_bits
        elif map_n == 22:
            return self.route11_bits
        elif map_n == 23:
            return self.route12_bits
        elif map_n == 24:
            return self.route13_bits
        elif map_n == 25:
            return self.route14_bits
        elif map_n == 26:
            return self.route15_bits
        elif map_n == 27:
            return self.route16_bits
        elif map_n == 28:
            return self.route17_bits
        elif map_n == 29:
            return self.route18_bits
        elif map_n == 30:
            return self.route19_bits
        elif map_n == 31:
            return self.route20_bits
        elif map_n == 32:
            return self.route21_bits
        elif map_n == 35:
            return self.route24_bits
        elif map_n == 36:
            return self.route25_bits
        elif map_n in self.poketower:
            return self.tower_bits
        elif map_n in self.pokehideout:
            return self.hideout_bits
        elif map_n in self.silphco:
            return self.silphco_bits
        elif map_n in self.rock_tunnel:
            return self.rock_tunnel_bits
        elif map_n in self.ssanne:
            return self.ssanne_bits
        elif map_n in self.mtmoon:
            return self.mtmoon_bits
        elif map_n in self.mansion:
            return self.mansion_bits
        elif map_n == 88:
            return self.bill_bits
        elif map_n == 51:
            return self.forest_bits
        elif map_n == 177:
            return self.dojo_bits
        elif map_n == 54:
            return self.brock_bits
        elif map_n == 65:
            return self.misty_bits
        elif map_n == 92:
            return self.surge_bits
        elif map_n == 134:
            return self.erika_bits
        elif map_n == 157:
            return self.koga_bits
        elif map_n == 178:
            return self.sabrina_bits
        elif map_n == 166:
            return self.blaine_bits
        elif map_n == 45:
            return self.giovanni_bits
        else:
            return self.misc_bits

    def event_rewards(self):
        current_dict = self.pick_dict()
        pop_list = []
        for key, value in current_dict.items():
            reward = self.calculate_reward(*value)
            if reward != 0:
                self.reward_calc += reward
                pop_list.append(key)
        for key in pop_list:
            current_dict.pop(key)
        return self.reward_calc

