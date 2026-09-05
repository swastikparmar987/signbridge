/**
 * SignBridge NLP Engine
 * Translates raw ASL gloss sequences into fluent, grammatically natural English sentences,
 * and extracts corresponding ASL glosses from spoken English for two-way communication.
 */

(function(window) {
  'use strict';

  // Irregular past tense mapping
  const PAST_VERBS = {
    'go': 'went',
    'eat': 'ate',
    'see': 'saw',
    'buy': 'bought',
    'drink': 'drank',
    'have': 'had',
    'make': 'made',
    'find': 'found',
    'take': 'took',
    'know': 'knew',
    'think': 'thought',
    'come': 'came',
    'give': 'gave',
    'read': 'read',
    'write': 'wrote',
    'meet': 'met',
    'teach': 'taught',
    'learn': 'learned',
    'sleep': 'slept',
    'break': 'broke',
    'tell': 'told',
    'hear': 'heard',
    'say': 'said',
    'feel': 'felt',
    'leave': 'left',
    'put': 'put',
    'bring': 'brought',
    'begin': 'began',
    'run': 'ran',
    'sit': 'sat',
    'stand': 'stood',
    'win': 'won',
    'lose': 'lost',
    'pay': 'paid',
    'send': 'sent',
    'build': 'built',
    'understand': 'understood'
  };

  // English lemmatization mapping for Speech-to-Text (spoken english -> ASL gloss base)
  const ENGLISH_TO_GLOSS_MAP = {
    'eating': 'EAT', 'ate': 'EAT', 'eats': 'EAT',
    'going': 'GO', 'went': 'GO', 'goes': 'GO',
    'seeing': 'SEE', 'saw': 'SEE', 'sees': 'SEE',
    'wanting': 'WANT', 'wants': 'WANT', 'wanted': 'WANT',
    'liking': 'LIKE', 'likes': 'LIKE', 'liked': 'LIKE',
    'drinking': 'DRINK', 'drank': 'DRINK', 'drinks': 'DRINK',
    'apples': 'APPLE', 'bananas': 'BANANA', 'books': 'BOOK',
    'cars': 'CAR', 'cats': 'CAT', 'dogs': 'DOG',
    'friends': 'FRIEND', 'families': 'FAMILY', 'labels': 'LABEL',
    'helping': 'HELP', 'helped': 'HELP', 'helps': 'HELP',
    'running': 'RUN', 'ran': 'RUN', 'runs': 'RUN',
    'playing': 'PLAY', 'played': 'PLAY', 'plays': 'PLAY',
    'working': 'WORK', 'worked': 'WORK', 'works': 'WORK',
    'learning': 'LEARN', 'learned': 'LEARN', 'learns': 'LEARN',
    'teaching': 'TEACH', 'taught': 'TEACH', 'teaches': 'TEACH',
    'thanks': 'THANKYOU', 'thank': 'THANKYOU',
    'sorry': 'SORRY', 'hello': 'HELLO', 'hi': 'HELLO', 'hey': 'HELLO',
    'goodbye': 'GOODBYE', 'bye': 'GOODBYE', 'please': 'PLEASE',
    'yes': 'YES', 'yeah': 'YES', 'yep': 'YES',
    'no': 'NO', 'nope': 'NO'
  };

  // Common idioms / fixed phrases
  const IDIOMS = [
    { pattern: ['NICE', 'MEET', 'YOU'], text: 'Nice to meet you!' },
    { pattern: ['ME', 'MEET', 'YOU', 'NICE'], text: 'It is nice to meet you!' },
    { pattern: ['SEE', 'YOU', 'LATER'], text: 'See you later!' },
    { pattern: ['GOOD', 'MORNING'], text: 'Good morning!' },
    { pattern: ['GOOD', 'NIGHT'], text: 'Good night!' },
    { pattern: ['GOOD', 'AFTERNOON'], text: 'Good afternoon!' },
    { pattern: ['HOW', 'YOU'], text: 'How are you?' },
    { pattern: ['YOU', 'HOW'], text: 'How are you doing?' },
    { pattern: ['NAME', 'YOU', 'WHAT'], text: 'What is your name?' },
    { pattern: ['YOU', 'NAME', 'WHAT'], text: 'What is your name?' },
    { pattern: ['MY', 'NAME'], text: 'My name is...' },
    { pattern: ['TIME', 'WHAT'], text: 'What time is it?' },
    { pattern: ['WHAT', 'TIME'], text: 'What time is it?' },
    { pattern: ['WHERE', 'BATHROOM'], text: 'Where is the restroom?' },
    { pattern: ['BATHROOM', 'WHERE'], text: 'Where is the bathroom?' },
    { pattern: ['TOILET', 'WHERE'], text: 'Where is the restroom?' },
    { pattern: ['HELP', 'ME'], text: 'Can you please help me?' },
    { pattern: ['ME', 'NEED', 'HELP'], text: 'I need help, please.' },
    { pattern: ['THANKYOU'], text: 'Thank you very much!' },
    { pattern: ['THANK', 'YOU'], text: 'Thank you very much!' },
    { pattern: ['WELCOME'], text: "You're welcome!" },
    { pattern: ['SORRY'], text: 'I am sorry.' },
    { pattern: ['PLEASE'], text: 'Please.' },
    { pattern: ['HELLO'], text: 'Hello!' },
    { pattern: ['YES'], text: 'Yes, absolutely.' },
    { pattern: ['NO'], text: 'No, thank you.' }
  ];

  const PRONOUNS = {
    'ME': 'I',
    'I': 'I',
    'MY': 'my',
    'MINE': 'mine',
    'YOU': 'you',
    'YOUR': 'your',
    'YOURS': 'yours',
    'HE': 'he',
    'SHE': 'she',
    'IT': 'it',
    'THIS/IT': 'this',
    'THIS': 'this',
    'THAT': 'that',
    'WE': 'we',
    'OUR': 'our',
    'THEY': 'they',
    'THEIR': 'their'
  };

  const TIME_PAST = new Set(['YESTERDAY', 'PAST', 'BEFORE', 'AGO', 'LAST-NIGHT']);
  const TIME_FUTURE = new Set(['TOMORROW', 'FUTURE', 'SOON', 'LATER', 'NEXT-WEEK']);
  const QUESTION_WORDS = new Set(['WHAT', 'WHERE', 'WHO', 'WHY', 'WHEN', 'HOW', 'WHICH']);

  function cleanGloss(gloss) {
    if (!gloss) return '';
    // Strip trailing numbers like APPLE1, WANT2, GO3
    return gloss.toUpperCase().replace(/\d+$/, '').replace(/\/IT$/, '');
  }

  function getArticle(word) {
    if (!word) return 'a';
    const firstChar = word.trim().charAt(0).toLowerCase();
    if (['a', 'e', 'i', 'o', 'u'].includes(firstChar)) {
      return 'an';
    }
    return 'a';
  }

  /**
   * Main NLP Gloss -> Fluent English Generator
   */
  function glossesToEnglish(rawGlosses) {
    if (!rawGlosses || rawGlosses.length === 0) {
      return '';
    }

    const cleaned = rawGlosses.map(g => cleanGloss(g)).filter(Boolean);
    if (cleaned.length === 0) return '';

    // Check fixed idioms / conversational patterns first
    for (const idiom of IDIOMS) {
      if (idiom.pattern.length === cleaned.length) {
        const matches = idiom.pattern.every((pat, idx) => pat === cleaned[idx]);
        if (matches) return idiom.text;
      }
    }

    // Check partial idiom matching
    const joinedClean = cleaned.join(' ');
    for (const idiom of IDIOMS) {
      const patJoined = idiom.pattern.join(' ');
      if (joinedClean === patJoined) return idiom.text;
    }

    // Check for Question formation
    const lastWord = cleaned[cleaned.length - 1];

    if (QUESTION_WORDS.has(lastWord)) {
      const qWord = lastWord.toLowerCase();
      const subjectTokens = cleaned.slice(0, -1);

      if (qWord === 'what') {
        if (subjectTokens.includes('NAME')) return 'What is your name?';
        if (subjectTokens.includes('TIME')) return 'What time is it?';
        if (subjectTokens.length > 0) {
          const sub = subjectTokens.map(t => PRONOUNS[t] || t.toLowerCase()).join(' ');
          return `What is ${sub}?`;
        }
        return 'What is that?';
      }

      if (qWord === 'where') {
        if (subjectTokens.length > 0) {
          const sub = subjectTokens.map(t => PRONOUNS[t] || t.toLowerCase()).join(' ');
          if (sub === 'you' || sub.includes('you go')) return 'Where are you going?';
          return `Where is the ${sub}?`;
        }
        return 'Where is it?';
      }

      if (qWord === 'who') {
        if (subjectTokens.length > 0) {
          const sub = subjectTokens.map(t => PRONOUNS[t] || t.toLowerCase()).join(' ');
          return `Who is ${sub}?`;
        }
        return 'Who is that?';
      }

      if (qWord === 'why') {
        if (subjectTokens.length > 0) {
          const sub = subjectTokens.map(t => PRONOUNS[t] || t.toLowerCase()).join(' ');
          return `Why ${sub}?`;
        }
        return 'Why is that?';
      }

      if (qWord === 'how') {
        if (subjectTokens.includes('YOU')) return 'How are you?';
        return 'How is that?';
      }
    }

    // Determine tense
    let isPast = cleaned.some(w => TIME_PAST.has(w));
    let isFuture = cleaned.some(w => TIME_FUTURE.has(w));

    // Token analysis
    const filteredTokens = cleaned.filter(w => !TIME_PAST.has(w) && !TIME_FUTURE.has(w));
    if (filteredTokens.length === 0) {
      if (isPast) return 'That was in the past.';
      if (isFuture) return 'That will be in the future.';
      return '';
    }

    // Pattern: Subject + Want/Like/Need + Object
    // e.g. [ME, WANT, APPLE] -> "I would like an apple."
    let subject = null;
    let verb = null;
    let object = null;

    let tokens = [...filteredTokens];

    // Check if first token is pronoun
    if (PRONOUNS[tokens[0]]) {
      subject = PRONOUNS[tokens[0]];
      tokens.shift();
    } else {
      subject = 'I'; // Default signer perspective
    }

    if (tokens.length > 0) {
      verb = tokens[0].toLowerCase();
      tokens.shift();
    }

    if (tokens.length > 0) {
      object = tokens.map(t => t.toLowerCase()).join(' ');
    }

    // Grammatical assembly
    let result = '';

    if (verb === 'want') {
      if (object) {
        const article = getArticle(object);
        result = `${subject} would like ${article} ${object}.`;
      } else {
        result = `${subject} want that.`;
      }
    } else if (verb === 'like') {
      result = object ? `${subject} like ${object}.` : `${subject} like it.`;
    } else if (verb === 'need') {
      result = object ? `${subject} need ${object}.` : `${subject} need help.`;
    } else if (verb === 'have') {
      if (isPast) {
        result = object ? `${subject} had ${object}.` : `${subject} had it.`;
      } else {
        result = object ? `${subject} have ${object}.` : `${subject} have it.`;
      }
    } else if (verb === 'go') {
      if (isPast) {
        result = object ? `${subject} went to the ${object}.` : `${subject} went there.`;
      } else if (isFuture) {
        result = object ? `${subject} will go to the ${object}.` : `${subject} will go there.`;
      } else {
        result = object ? `${subject} am going to the ${object}.` : `${subject} am going.`;
      }
    } else if (verb === 'eat') {
      if (isPast) {
        result = object ? `${subject} ate ${getArticle(object)} ${object}.` : `${subject} ate.`;
      } else if (isFuture) {
        result = object ? `${subject} will eat ${getArticle(object)} ${object}.` : `${subject} will eat.`;
      } else {
        result = object ? `${subject} would like to eat ${getArticle(object)} ${object}.` : `${subject} am eating.`;
      }
    } else {
      // General token sequence
      const englishTokens = cleaned.map(w => {
        if (PRONOUNS[w]) return PRONOUNS[w];
        const lower = w.toLowerCase();
        if (isPast && PAST_VERBS[lower]) return PAST_VERBS[lower];
        return lower;
      });

      result = englishTokens.join(' ');
      if (isFuture && !result.toLowerCase().includes('will')) {
        result = 'will ' + result;
      }
      result = result.charAt(0).toUpperCase() + result.slice(1) + '.';
    }

    // Capitalize first character
    if (result) {
      result = result.charAt(0).toUpperCase() + result.slice(1);
    }

    return result;
  }

  /**
   * Speech-to-Text: Match spoken English against ASL model vocabulary
   */
  function extractASLGlossesFromSpeech(spokenText, availableGlossesSet) {
    if (!spokenText) return [];
    
    // Normalize and clean punctuation
    const words = spokenText.toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(Boolean);

    const matches = [];

    for (const word of words) {
      // 1. Direct match (e.g. "apple" -> "APPLE")
      const upper = word.toUpperCase();
      if (availableGlossesSet && availableGlossesSet.has(upper)) {
        matches.push({ word: word, gloss: upper, type: 'direct' });
        continue;
      }

      // 2. Lemmatized match (e.g. "eating" -> "EAT")
      if (ENGLISH_TO_GLOSS_MAP[word]) {
        const mappedGloss = ENGLISH_TO_GLOSS_MAP[word];
        if (!availableGlossesSet || availableGlossesSet.has(mappedGloss)) {
          matches.push({ word: word, gloss: mappedGloss, type: 'stemmed' });
          continue;
        }
      }

      // 3. Fallback suffix removal (-s, -ing, -ed)
      let stem = null;
      if (word.endsWith('ing') && word.length > 5) stem = word.slice(0, -3).toUpperCase();
      else if (word.endsWith('ed') && word.length > 4) stem = word.slice(0, -2).toUpperCase();
      else if (word.endsWith('s') && word.length > 3) stem = word.slice(0, -1).toUpperCase();

      if (stem && availableGlossesSet && availableGlossesSet.has(stem)) {
        matches.push({ word: word, gloss: stem, type: 'stemmed' });
      }
    }

    // Deduplicate by gloss
    const seen = new Set();
    return matches.filter(m => {
      if (seen.has(m.gloss)) return false;
      seen.add(m.gloss);
      return true;
    });
  }

  // Export to window
  window.SignBridgeNLP = {
    cleanGloss,
    glossesToEnglish,
    extractASLGlossesFromSpeech
  };

})(window);
