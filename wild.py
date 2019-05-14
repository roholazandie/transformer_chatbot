# from ParlAI.projects.convai.convai_world import ConvAIWorld
from parlai.core.params import ParlaiParser
from agent import TransformerAgent


def main():
    parser = ParlaiParser(True, True)
    parser.set_defaults(batchsize=10,
                        sample=True,
                        wild_mode=False,
                        replace_repeat=True,
                        replace_ngram=True,
                        detokenize=True,
                        emoji_prob=0.0,
                        add_questions=0.4,
                        clean_emoji=True,
                        check_grammar=True,
                        correct_generative=True,
                        split_into_sentences=True,
                        max_seq_len=256,
                        beam_size=3,
                        annealing_topk=None,
                        annealing=0.6,
                        length_penalty=0.7)

    # ConvAIWorld.add_cmdline_args(parser)
    TransformerAgent.add_cmdline_args(parser)
    opt = parser.parse_args()

    agent = TransformerAgent(opt)

    '''
    Two scenarios for running the model in eval mode:
    1- with persona and information of dialog then the long text below will be useful(should be replaced with "I am good. How are you?")
    2- without persona just like below. text can be the input. 
    '''

    text = "your persona: i love to redesign houses. \n " \
           "your persona: killing for sport is my hobby. \n" \
           "your persona: i shot an arrow the other day !.\n" \
           "your persona: i like to get dressed up.\n" \
           "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .	you must be very fast . hunting is one of my favorite hobbies .\n" \
           "i am ! for my hobby i like to do canning or some whittling .	i also remodel homes when i am not out bow hunting .\n" \
           "that's neat . when i was in high school i placed 6th in 100m dash !	that's awesome . do you have a favorite season or time of year ?"

    while True:
        user_text = input(">")
        input_var = {
            'id': 'MasterBot#%s' % 0,
            'text': user_text,  # "I am good. How are you?",
            'episode_done': False
        }

        observation = agent.observe(input_var)
        response = agent.act()
        print(response['text'])


if __name__ == '__main__':
    main()
