## General instructions: 
1. The app code is structured in two main directories: app & client
2. The app holds the backend & the client is the frontend. 

## Instructions on writing code
1. Please be conservative in the changes that you are doing. In most cases, I will be telling you exactly what I want you to do.
2. Usually, it is not a good idea to create new files, class, model, interface, type definitons. As much as possible, you should use existing ones. If you think one is really needed or recommended, check with me once before you proceed. Do take heed of this, creating these new assets without explicit verbal permission from me is UNACCEPTABLE.
3. In case of any confusion, ambuiguity or inconsistencies in instructions, it's usually best to ask me rather than make assumptions & do something very adventurous.
4. I couldn't emphasize this enough. Please ask ask ask before assuming anything. Question every assumption of yours. This will ensure that the code remains free of slop/fluff and it will ensure that every new addition to the code is well thought of (even by me when you ask me questions).
5. When generating Rust code refer to the Rust styling guide available in docs/rust_guidelines.md file.
6. When generating Typescript code refer to the Typescript styling guide available in docs/typescript_guidelines.md file.

## Shared type definitions between the Rust backend and the Typescript frontend
1. Use typeshare to generate Typescript definitions from Rust struct definitions using the following command in the project root directory: 
```bash 
`typeshare app/src/ --lang=typescript --output-file=client/src/common/types.ts` 
```

## Structured feature implementation
1. Before writing any code for a structured feature, write the high-level spec to `docs/specs/{feature}.md` and the implementation plan to `docs/specs/{feature}-impl.md`.