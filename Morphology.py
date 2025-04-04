from autocorrect import Speller

def spell_check_and_autocorrect(text):
 

  spell = Speller()
  corrected_text = ' '.join([spell(word) for word in text.split()])
  return corrected_text

# Example usage
input_text = "This is a texxt with some mispelled wrods."
corrected_text = spell_check_and_autocorrect(input_text)

print("Corrected Text:", corrected_text)
