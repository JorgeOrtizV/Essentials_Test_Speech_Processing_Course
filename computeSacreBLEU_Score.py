from sacrebleu.metrics import BLEU

if __name__ == "__main__":
    refs = [['The first to arrive at the finish line, after three hours and 13 minutes was Fred Lorz.'], 
    ['After being hailed as the winner, he had his photograph taken with Alice Roosevelt, daughter of then-U.S. President Theodore Roosevelt, and was about to be awarded the gold medal when his subterfuge was revealed.'],
    ['Lorz had actually dropped out of the race after nine miles after suffering cramps and hitched a ride back to the stadium in a car, waving at spectators and runners alike during the ride.'],
    ['When the car broke down at the 19th mile, he re-entered the race and jogged across the finish line.'],
    ['Upon being confronted by officials, Lorz immediately admitted his deception, and despite his claims he was joking, the AAU responded by banning him for life.']]

    sys = ['Fred Lorz was the first to cross the finish line after three hours and 13 minutes.',
    'After being celebrated as the winner, he had his picture taken with Alice Roosevelt, daughter of then U.S. President Theodore Roosevelt, and was about to accept the gold medal when his deception was exposed.',
    'In reality, Lorz had abandoned the race after nine miles after suffering cramps and had ridden in a car back to the stadium, waving to spectators and runners alike.',
    'When the car broke down at the 19th mile, he resumed the race and jogged across the finish line.',
    'When confronted by officials, Lorz immediately admitted his cheating, and despite his claims that he was only joking, the AAU responded with a lifetime ban.']

    bleu = BLEU()

    print(bleu.corpus_score(sys, refs))
