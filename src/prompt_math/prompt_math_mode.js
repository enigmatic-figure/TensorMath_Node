// CodeMirror mode for Prompt-Math syntax highlighting
(function(mod) {
    if (typeof exports == "object" && typeof module == "object") // CommonJS
        mod(require("../../lib/codemirror"));
    else if (typeof define == "function" && define.amd) // AMD
        define(["../../lib/codemirror"], mod);
    else // Plain browser env
        mod(CodeMirror);
})(function(CodeMirror) {
    "use strict";

    // Define the Prompt-Math mode
    CodeMirror.defineMode("promptmath", function(config, parserConfig) {
        
        // Token types for syntax highlighting
        const TOKEN_TYPES = {
            BRACKET_DOUBLE: "prompt-math-bracket",
            BRACKET_SINGLE: "prompt-math-bracket", 
            TOKEN: "prompt-math-token",
            FUNCTION: "prompt-math-function",
            SCHEDULE: "prompt-math-schedule",
            NUMBER: "prompt-math-number",
            OPERATOR: "prompt-math-operator",
            INVALID: "prompt-math-invalid",
            COMMENT: "comment",
            STRING: "string"
        };

        // Built-in functions
        const MATH_FUNCTIONS = new Set([
            'mean', 'unit', 'pow', 'clipnorm', 'match_stats', 'delta', 'proj',
            'lerp', 'slerp', 'ortho', 'pca_reduce', 'mask', 'normalize',
            'clamp', 'abs', 'sign', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tanh'
        ]);

        const SCHEDULE_FUNCTIONS = new Set([
            'fade_in', 'fade_out', 'emphasis', 'schedule', 'bell', 'pulse',
            'step', 'linear', 'smooth', 'ease_in', 'ease_out', 'exponential',
            'logarithmic', 'constant', 'ramp', 'triangle', 'sawtooth'
        ]);

        const CURVE_TYPES = new Set([
            'linear', 'smooth', 'smoothstep', 'ease_in', 'easein', 'ease_out', 
            'easeout', 'bell', 'gaussian', 'step', 'threshold', 'pulse', 'sine',
            'exp', 'exponential', 'log', 'logarithmic'
        ]);

        // State object for the parser
        function State() {
            return {
                inDoubleExpr: false,
                inToken: false,
                inSchedule: false,
                bracketDepth: 0,
                parenDepth: 0,
                braceDepth: 0,
                lastToken: null,
                expectingSchedule: false
            };
        }

        // Token function - main parsing logic
        function token(stream, state) {
            // Skip whitespace
            if (stream.eatSpace()) {
                return null;
            }

            // Comments (# to end of line)
            if (stream.match(/^#.*$/)) {
                return TOKEN_TYPES.COMMENT;
            }

            // Double bracket expressions [[ ... ]]
            if (stream.match(/^\[\[/)) {
                state.inDoubleExpr = true;
                state.bracketDepth += 2;
                return TOKEN_TYPES.BRACKET_DOUBLE;
            }

            if (state.inDoubleExpr && stream.match(/^\]\]/)) {
                state.inDoubleExpr = false;
                state.bracketDepth -= 2;
                return TOKEN_TYPES.BRACKET_DOUBLE;
            }

            // Single bracket tokens [ ... ]
            if (stream.match(/^\[/)) {
                state.inToken = true;
                state.bracketDepth += 1;
                return TOKEN_TYPES.BRACKET_SINGLE;
            }

            if (state.inToken && stream.match(/^\]/)) {
                state.inToken = false;
                state.bracketDepth -= 1;
                return TOKEN_TYPES.BRACKET_SINGLE;
            }

            // Braces for slicing and guards { ... }
            if (stream.match(/^\{/)) {
                state.braceDepth += 1;
                return TOKEN_TYPES.BRACKET_SINGLE;
            }

            if (stream.match(/^\}/)) {
                state.braceDepth -= 1;
                return TOKEN_TYPES.BRACKET_SINGLE;
            }

            // Parentheses
            if (stream.match(/^\(/)) {
                state.parenDepth += 1;
                return TOKEN_TYPES.OPERATOR;
            }

            if (stream.match(/^\)/)) {
                state.parenDepth -= 1;
                return TOKEN_TYPES.OPERATOR;
            }

            // Schedule operator @
            if (stream.match(/^@/)) {
                state.expectingSchedule = true;
                return TOKEN_TYPES.SCHEDULE;
            }

            // Arrow operator ->
            if (stream.match(/^->/)) {
                return TOKEN_TYPES.OPERATOR;
            }

            // Mathematical operators
            if (stream.match(/^[+\-*/]/)) {
                return TOKEN_TYPES.OPERATOR;
            }

            // Comparison operators
            if (stream.match(/^[<>=]/)) {
                return TOKEN_TYPES.OPERATOR;
            }

            // Special symbols
            if (stream.match(/^[,:;|∉]/)) {
                return TOKEN_TYPES.OPERATOR;
            }

            // Numbers (including floats)
            if (stream.match(/^\d+(\.\d+)?/)) {
                return TOKEN_TYPES.NUMBER;
            }

            // Quoted strings
            if (stream.match(/^"([^"\\]|\\.)*"/)) {
                return TOKEN_TYPES.STRING;
            }

            // Identifiers (functions, tokens, etc.)
            if (stream.match(/^[a-zA-Z_][a-zA-Z0-9_]*/)) {
                const word = stream.current();
                
                // Check if it's a function call (followed by parentheses)
                const nextChar = stream.peek();
                if (nextChar === '(') {
                    if (state.expectingSchedule && SCHEDULE_FUNCTIONS.has(word)) {
                        state.expectingSchedule = false;
                        return TOKEN_TYPES.SCHEDULE;
                    } else if (MATH_FUNCTIONS.has(word)) {
                        return TOKEN_TYPES.FUNCTION;
                    } else {
                        // Unknown function
                        return TOKEN_TYPES.FUNCTION;
                    }
                }

                // Check for curve types after |
                if (state.lastToken === '|' && CURVE_TYPES.has(word)) {
                    return TOKEN_TYPES.SCHEDULE;
                }

                // Inside token brackets, treat as token content
                if (state.inToken) {
                    return TOKEN_TYPES.TOKEN;
                }

                // Default identifier
                return null;
            }

            // Token content inside brackets (including spaces and special chars)
            if (state.inToken && !stream.match(/^[\[\]]/)) {
                stream.next();
                return TOKEN_TYPES.TOKEN;
            }

            // Consume any other character
            stream.next();
            return null;
        }

        // Start state function
        function startState() {
            return new State();
        }

        // Copy state function for undo/redo
        function copyState(state) {
            return {
                inDoubleExpr: state.inDoubleExpr,
                inToken: state.inToken,
                inSchedule: state.inSchedule,
                bracketDepth: state.bracketDepth,
                parenDepth: state.parenDepth,
                braceDepth: state.braceDepth,
                lastToken: state.lastToken,
                expectingSchedule: state.expectingSchedule
            };
        }

        // Indentation function
        function indent(state, textAfter) {
            const unit = config.indentUnit || 2;
            let indent = 0;
            
            // Indent based on bracket depth
            indent += state.bracketDepth * unit;
            indent += state.parenDepth * unit;
            
            return indent;
        }

        // Electric characters that trigger re-indentation
        const electricChars = "[](){}";

        // Return the mode object
        return {
            startState: startState,
            copyState: copyState,
            token: token,
            indent: indent,
            electricChars: electricChars,
            lineComment: "#",
            fold: "brace"
        };
    });

    // Define MIME type
    CodeMirror.defineMIME("text/x-promptmath", "promptmath");

    // Syntax validation function
    CodeMirror.registerHelper("lint", "promptmath", function(text) {
        const errors = [];
        const lines = text.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNum = i + 1;
            
            // Check for unmatched brackets
            const bracketStack = [];
            const bracketPairs = {
                '[': ']',
                '(': ')',
                '{': '}'
            };
            
            for (let j = 0; j < line.length; j++) {
                const char = line[j];
                
                if (char in bracketPairs) {
                    bracketStack.push({
                        char: char,
                        pos: j,
                        expected: bracketPairs[char]
                    });
                } else if (Object.values(bracketPairs).includes(char)) {
                    if (bracketStack.length === 0) {
                        errors.push({
                            message: `Unmatched closing bracket '${char}'`,
                            severity: "error",
                            from: CodeMirror.Pos(i, j),
                            to: CodeMirror.Pos(i, j + 1)
                        });
                    } else {
                        const last = bracketStack.pop();
                        if (last.expected !== char) {
                            errors.push({
                                message: `Expected '${last.expected}' but found '${char}'`,
                                severity: "error",
                                from: CodeMirror.Pos(i, j),
                                to: CodeMirror.Pos(i, j + 1)
                            });
                        }
                    }
                }
            }
            
            // Check for unclosed brackets at end of line
            for (const bracket of bracketStack) {
                errors.push({
                    message: `Unclosed bracket '${bracket.char}'`,
                    severity: "error",
                    from: CodeMirror.Pos(i, bracket.pos),
                    to: CodeMirror.Pos(i, bracket.pos + 1)
                });
            }
            
            // Check for invalid double bracket expressions
            const doubleBracketMatches = line.match(/\[\[|\]\]/g);
            if (doubleBracketMatches && doubleBracketMatches.length % 2 !== 0) {
                errors.push({
                    message: "Unmatched double brackets [[ ]]",
                    severity: "error",
                    from: CodeMirror.Pos(i, 0),
                    to: CodeMirror.Pos(i, line.length)
                });
            }
            
            // Check for invalid schedule syntax
            const scheduleMatches = line.match(/@\s*([a-zA-Z_][a-zA-Z0-9_]*)/g);
            if (scheduleMatches) {
                for (const match of scheduleMatches) {
                    const funcName = match.replace('@', '').trim();
                    const schedFuncs = [
                        'fade_in', 'fade_out', 'emphasis', 'schedule', 'bell', 
                        'pulse', 'step', 'constant', 'linear', 'smooth'
                    ];
                    
                    if (!schedFuncs.includes(funcName)) {
                        const pos = line.indexOf(match);
                        errors.push({
                            message: `Unknown schedule function: ${funcName}`,
                            severity: "warning",
                            from: CodeMirror.Pos(i, pos),
                            to: CodeMirror.Pos(i, pos + match.length)
                        });
                    }
                }
            }
        }
        
        return errors;
    });

    // Auto-completion hints
    CodeMirror.registerHelper("hint", "promptmath", function(editor, options) {
        const cursor = editor.getCursor();
        const line = editor.getLine(cursor.line);
        const start = cursor.ch;
        let end = cursor.ch;
        
        // Find the word being typed
        while (end < line.length && /[\w$]/.test(line.charAt(end))) ++end;
        while (start && /[\w$]/.test(line.charAt(start - 1))) --start;
        
        const word = line.slice(start, end);
        
        // Build completion list
        const completions = [];
        
        // Mathematical functions
        const mathFuncs = [
            'mean(', 'unit(', 'pow(', 'clipnorm(', 'match_stats(', 'delta(',
            'lerp(', 'slerp(', 'ortho(', 'normalize(', 'clamp(', 'abs(', 'sqrt('
        ];
        
        // Schedule functions
        const schedFuncs = [
            'fade_in(', 'fade_out(', 'emphasis(', 'schedule(', 'bell(',
            'pulse(', 'step(', 'constant(', 'linear', 'smooth', 'ease_in', 'ease_out'
        ];
        
        // Check context to determine what to suggest
        const beforeCursor = line.slice(0, start);
        const isAfterAt = /@\s*$/.test(beforeCursor);
        const isAfterPipe = /\|\s*$/.test(beforeCursor);
        
        if (isAfterAt) {
            // Suggest schedule functions after @
            for (const func of schedFuncs) {
                if (func.startsWith(word)) {
                    completions.push({
                        text: func,
                        displayText: func,
                        className: "cm-hint-schedule"
                    });
                }
            }
        } else if (isAfterPipe) {
            // Suggest curve types after |
            const curves = ['linear', 'smooth', 'ease_in', 'ease_out', 'bell', 'step', 'pulse'];
            for (const curve of curves) {
                if (curve.startsWith(word)) {
                    completions.push({
                        text: curve,
                        displayText: curve,
                        className: "cm-hint-curve"
                    });
                }
            }
        } else {
            // Suggest math functions
            for (const func of mathFuncs) {
                if (func.startsWith(word)) {
                    completions.push({
                        text: func,
                        displayText: func,
                        className: "cm-hint-function"
                    });
                }
            }
        }
        
        // Common tokens
        const commonTokens = [
            '[king]', '[man]', '[woman]', '[detailed]', '[sharp]', '[blurry]',
            '[oil_painting]', '[watercolor]', '[photo]', '[realistic]'
        ];
        
        for (const token of commonTokens) {
            if (token.toLowerCase().includes(word.toLowerCase())) {
                completions.push({
                    text: token,
                    displayText: token,
                    className: "cm-hint-token"
                });
            }
        }
        
        return {
            list: completions,
            from: CodeMirror.Pos(cursor.line, start),
            to: CodeMirror.Pos(cursor.line, end)
        };
    });

    // Bracket matching configuration
    CodeMirror.defineOption("matchBrackets", false, function(cm, val, old) {
        if (old && old != CodeMirror.Init) {
            cm.off("cursorActivity", matchBrackets);
            cm.off("focus", matchBrackets);
            cm.off("blur", clearMatched);
            clearMatched(cm);
        }
        if (val) {
            cm.state.matchBrackets = typeof val == "object" ? val : {};
            cm.on("cursorActivity", matchBrackets);
            cm.on("focus", matchBrackets);
            cm.on("blur", clearMatched);
        }
    });

    function matchBrackets(cm) {
        cm.operation(function() {
            const marks = cm.state.matchBrackets.marks || [];
            for (let i = 0; i < marks.length; i++) {
                marks[i].clear();
            }
            marks.length = 0;

            const cursor = cm.getCursor();
            const line = cm.getLine(cursor.line);
            const pos = cursor.ch;

            // Check for bracket at cursor position or adjacent
            const brackets = "[](){}";
            let bracket = null;
            let bracketPos = null;

            if (pos > 0 && brackets.includes(line[pos - 1])) {
                bracket = line[pos - 1];
                bracketPos = pos - 1;
            } else if (pos < line.length && brackets.includes(line[pos])) {
                bracket = line[pos];
                bracketPos = pos;
            }

            if (bracket) {
                const match = findMatchingBracket(cm, cursor.line, bracketPos, bracket);
                if (match) {
                    const mark1 = cm.markText(
                        {line: cursor.line, ch: bracketPos},
                        {line: cursor.line, ch: bracketPos + 1},
                        {className: "CodeMirror-matchingbracket"}
                    );
                    const mark2 = cm.markText(
                        {line: match.line, ch: match.ch},
                        {line: match.line, ch: match.ch + 1},
                        {className: "CodeMirror-matchingbracket"}
                    );
                    marks.push(mark1, mark2);
                } else {
                    const mark = cm.markText(
                        {line: cursor.line, ch: bracketPos},
                        {line: cursor.line, ch: bracketPos + 1},
                        {className: "CodeMirror-nonmatchingbracket"}
                    );
                    marks.push(mark);
                }
            }

            cm.state.matchBrackets.marks = marks;
        });
    }

    function findMatchingBracket(cm, line, pos, bracket) {
        const pairs = {
            '(': ')', '[': ']', '{': '}',
            ')': '(', ']': '[', '}': '{'
        };
        
        const match = pairs[bracket];
        if (!match) return null;

        const isOpening = '([{'.includes(bracket);
        const direction = isOpening ? 1 : -1;
        let depth = 1;

        const maxLines = cm.lineCount();
        let currentLine = line;
        let currentPos = pos + direction;

        while (currentLine >= 0 && currentLine < maxLines) {
            const lineText = cm.getLine(currentLine);
            
            if (direction === 1) {
                for (let i = currentPos; i < lineText.length; i++) {
                    const char = lineText[i];
                    if (char === bracket) {
                        depth++;
                    } else if (char === match) {
                        depth--;
                        if (depth === 0) {
                            return {line: currentLine, ch: i};
                        }
                    }
                }
            } else {
                for (let i = currentPos; i >= 0; i--) {
                    const char = lineText[i];
                    if (char === bracket) {
                        depth++;
                    } else if (char === match) {
                        depth--;
                        if (depth === 0) {
                            return {line: currentLine, ch: i};
                        }
                    }
                }
            }

            currentLine += direction;
            currentPos = direction === 1 ? 0 : (currentLine >= 0 && currentLine < maxLines ? cm.getLine(currentLine).length - 1 : 0);
        }

        return null;
    }

    function clearMatched(cm) {
        const marks = cm.state.matchBrackets && cm.state.matchBrackets.marks;
        if (marks) {
            for (let i = 0; i < marks.length; i++) {
                marks[i].clear();
            }
            marks.length = 0;
        }
    }
});