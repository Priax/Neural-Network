##
## EPITECH PROJECT, 2024
## Makefile
## File description:
## my_pgp
##

ANALYZER = my_torch_analyzer
SRC = neural_network_analyzer.py

GENERATOR = my_torch_generator
SRC2 = neural_network_generator.py

all: $(ANALYZER) $(GENERATOR)

$(ANALYZER): $(SRC)
	cp $(SRC) $(ANALYZER)
	chmod 777 $(ANALYZER)

$(GENERATOR): $(SRC2)
	cp $(SRC2) $(GENERATOR)
	chmod 777 $(GENERATOR)

clean:
	rm -f *.o

fclean: clean
	rm -f $(ANALYZER)
	rm -f $(GENERATOR)

re: fclean all
