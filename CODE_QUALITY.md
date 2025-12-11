# TypeScript/Next.js Code Quality Guidelines

This document defines **non-negotiable guardrails** for TypeScript code in this project. The goal is to make the codebase:

* Easy to understand on first read
* Easy to change without fear
* Hard to break accidentally

Assume many engineers will cycle through this project. Your job when writing code is to be a **good ancestor**.

---

## 1. Core Philosophy

1. **Correctness first**

   * A small correct solution beats a clever incomplete one.
   * We prefer obvious, boring code over "smart" tricks.

2. **Functional at the core**

   * Data in → data out, with minimal side effects.
   * Side effects are isolated at the edges (I/O, network, UI event handlers).

3. **Strict typing**

   * All interfaces are fully typed with no escape hatches.
   * Type errors are build failures. Zero tolerance for `any`.

4. **Immutability by default**

   * Prefer `readonly`, `as const`, and immutable update patterns.
   * Mutation is exceptional and must be explicitly justified.

5. **Low lexical and structural complexity**

   * Small functions, shallow nesting, short modules.
   * No "god components", no "god hooks".

6. **Idiomatic TypeScript/React**

   * Use the common patterns the community expects.
   * Avoid re-implementing standard library or React features.

7. **Consistency over preference**

   * Follow project conventions even if you disagree with them.
   * If you want to change a convention, propose it; don't fork it.

---

## 2. Language Subset and Style

### 2.1 Allowed language features

* TypeScript strict mode with all strict flags enabled.
* Use `interface` for object shapes, `type` for unions/intersections/utilities.
* Use `const` by default; `let` only when reassignment is necessary.
* Use `readonly` modifiers on all interface properties unless mutation is required.
* Use discriminated unions for state machines and variant types.
* Use template literal types for string validation where appropriate.

### 2.2 Discouraged / forbidden patterns

* `any` — use `unknown` and narrow with type guards.
* `as` type assertions — use type guards or refine the type system instead.
* `!` non-null assertions — handle null/undefined explicitly.
* `enum` — use `as const` objects or union types instead.
* `class` for React components — use function components exclusively.
* Deep inheritance hierarchies — prefer composition.
* Global mutable state outside of explicit state management.
* `var` declarations.
* Implicit type coercion; be explicit with conversions.

---

## 3. Immutability Patterns

Immutability is the default. Mutation is opt-in and requires justification.

### 3.1 Interface design

```typescript
// Preferred: readonly by default
interface User {
  readonly id: string;
  readonly name: string;
  readonly email: string;
}

// For deeply immutable structures
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};
```

### 3.2 Array and object operations

* Prefer spread operators over mutating methods:

  * `[...arr, newItem]` over `arr.push(newItem)`
  * `{ ...obj, key: newValue }` over `obj.key = newValue`
  * `arr.filter(x => x.id !== id)` over `arr.splice(index, 1)`

* Use `map`, `filter`, `reduce` over imperative loops with mutation.

* For complex updates, use `structuredClone()` or a library like Immer.

### 3.3 React state

* Never mutate state directly; always use setter functions with new references.
* For complex state, prefer `useReducer` with immutable update patterns.
* Consider Immer's `produce` for deeply nested state updates.

---

## 4. Functional Paradigms

We write functional-first TypeScript. Side effects live at the edges.

### 4.1 Design

* Prefer:

  * Small, pure functions that transform data.
  * Functions that receive all inputs via parameters.
  * Functions that return new data instead of mutating arguments.

* Avoid:

  * Functions that implicitly read from or write to global state.
  * Functions that both compute and perform I/O; separate orchestration from logic.

### 4.2 Side-effect boundaries

* Centralize side effects in clearly named layers:

  * `lib/api/` or `services/` for network calls.
  * `hooks/` for React side effects (useEffect, mutations).
  * Event handlers at component boundaries.

* Rule of thumb:

  * Pure functions should be trivially unit testable.
  * If a function is hard to test without mocking, it likely mixes concerns.

### 4.3 Component composition

* Prefer composition over prop drilling.
* Use render props or compound components for flexible APIs.
* Keep components small and focused on a single responsibility.

---

## 5. Strict Typing

Static typing is mandatory, not optional.

### 5.1 Requirements

* Every function:

  * Has explicit return type annotations (don't rely on inference for public APIs).
  * Uses typed parameters; no implicit `any`.

* All component props are typed via interfaces.

* All hook return types are explicit.

### 5.2 TypeScript configuration

```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true
  }
}
```

### 5.3 Type design rules

* Prefer:

  * Narrow, precise types over `unknown` or wide unions.
  * Branded/opaque types for domain identifiers (UserId, OrderId).
  * Discriminated unions for state machines.

* Avoid:

  * `any`, `object`, or `Function` types.
  * Overly generic types; introduce generics only when actually reused.
  * Optional properties when a discriminated union is clearer.

### 5.4 Type guards and narrowing

```typescript
// Prefer type predicates for reusable narrowing
function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'name' in value
  );
}

// Use exhaustive checks for discriminated unions
function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${value}`);
}
```

---

## 6. Low Lexical and Structural Complexity

### 6.1 Function complexity limits

Per function/hook:

* Maximum cyclomatic complexity: 10
* Maximum nesting depth (if/for/switch/try): 3 levels
* Maximum function length: 40–50 lines
* Maximum number of parameters: 4

If you hit these limits:

* Split into smaller helper functions.
* Extract logic into custom hooks.
* Use early returns to flatten conditionals.

### 6.2 Component complexity limits

Per component:

* Maximum JSX depth: 4 levels of nesting
* Maximum hooks per component: 5
* Maximum props: 7 (consider composition if more)

If you hit these limits:

* Extract subcomponents.
* Create custom hooks for logic.
* Use composition patterns.

### 6.3 Module / file complexity limits

Per module/file:

* Maximum file length: ~300 lines (soft limit).
* One component per file (with small helper components as exceptions).
* Prefer small modules with cohesive responsibilities.

### 6.4 Lexical complexity rules

* Keep local variables minimal; prefer introducing small helpers.
* Avoid:

  * Long boolean expressions without intermediate naming.
  * Nested ternaries; use early returns or extracted functions.
  * Anonymous functions in JSX when they exceed one line.

* Use meaningful, full-word names (no `x1`, `tmp2`, etc.).

---

## 7. Idiomatic TypeScript/React

### 7.1 Naming conventions

* Files: `kebab-case.ts` or `PascalCase.tsx` for components.
* Interfaces/Types: `PascalCase` (e.g., `UserProfile`, `ApiResponse`).
* Functions/variables: `camelCase`.
* Constants: `SCREAMING_SNAKE_CASE` for true constants, `camelCase` for const references.
* React components: `PascalCase`.
* Hooks: `useCamelCase`.
* Event handlers: `handleEventName` or `onEventName`.

### 7.2 Control flow

* Prefer early returns to reduce nesting.
* Use optional chaining (`?.`) and nullish coalescing (`??`) appropriately.
* Avoid clever one-liners when they obscure logic.
* Use `switch` with exhaustive checks for discriminated unions.

### 7.3 React patterns

* Use function components exclusively.
* Prefer controlled components over uncontrolled.
* Use `key` props correctly (stable, unique identifiers, not array indices).
* Memoize appropriately: `useMemo` for expensive computations, `useCallback` for stable references.
* Don't over-memoize; measure before optimizing.

---

## 8. Next.js Patterns

### 8.1 App Router conventions

* Use Server Components by default; Client Components only when necessary.
* Mark client components explicitly with `'use client'` at the top.
* Prefer Server Actions for mutations over client-side API calls.
* Use `loading.tsx` and `error.tsx` for loading/error states.

### 8.2 Data fetching

* Fetch data in Server Components when possible.
* Use React Query or SWR for client-side data fetching with caching.
* Avoid `useEffect` for data fetching in Client Components.

### 8.3 File organization

```
src/
├── app/                 # Next.js App Router pages
├── components/          # Shared UI components
│   ├── ui/             # Primitive UI components
│   └── features/       # Feature-specific components
├── hooks/              # Custom React hooks
├── lib/                # Utilities and helpers
│   ├── api/           # API client functions
│   └── utils/         # Pure utility functions
├── types/              # Shared TypeScript types
└── stores/             # State management (if needed)
```

---

## 9. Error Handling

### 9.1 Error handling principles

* Fail fast on invalid state:

  * Validate inputs at boundaries (API responses, user input).
  * Use Zod or similar for runtime validation with type inference.

* Do not:

  * Silently swallow errors; at minimum, log with context.
  * Use exceptions for normal control flow.
  * Catch errors without handling them meaningfully.

### 9.2 Result types

Consider using Result types for operations that can fail:

```typescript
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };
```

### 9.3 Error boundaries

* Use React Error Boundaries for graceful degradation.
* Provide meaningful fallback UIs.
* Log errors to monitoring service.

---

## 10. Testing

### 10.1 Testing requirements

* Every new feature:

  * Unit tests for pure functions and hooks.
  * Component tests for UI behavior.
  * Integration tests for critical user flows.

* Rules:

  * Test names describe behavior, not implementation.
  * Tests must be deterministic; no reliance on timing.
  * Prefer Testing Library queries that reflect user behavior.

### 10.2 Testing patterns

```typescript
// Prefer: describe behavior
it('displays error message when form validation fails', () => {});

// Avoid: describe implementation
it('calls setError with validation message', () => {});
```

---

## 11. Pre-Build Quality Pipeline

Every commit must pass the following checks. The build fails if any step fails.

### 11.1 Code formatting

* Use Prettier with project configuration.
* No manual style bikeshedding; Prettier is the source of truth.
* Format on save; format on commit via hooks.

### 11.2 Linting

ESLint with TypeScript and React plugins:

* `@typescript-eslint/strict` rules enabled.
* `eslint-plugin-react-hooks` for hooks rules.
* `eslint-plugin-jsx-a11y` for accessibility.

Lint rules include:

* No unused variables or imports.
* No explicit `any`.
* Consistent import ordering.
* No console.log in production code.
* Exhaustive deps for hooks.

### 11.3 Type checking

* `tsc --noEmit` must pass with zero errors.
* No `@ts-ignore` or `@ts-expect-error` without:

  * Justification in a comment.
  * A ticket reference for known limitations.

### 11.4 Pre-commit workflow

Pre-commit hooks run:

* Prettier (format).
* ESLint (lint).
* TypeScript (type check on staged files).
* Tests (affected tests).

---

## 12. Code Review Standards

Code review is the last gate against complexity and bugs.

### 12.1 Reviewer checklist

For each change:

* **Correctness:**

  * Does the change do what the description claims?
  * Are edge cases handled?
  * Are null/undefined cases covered?

* **Types:**

  * Are all new functions fully typed?
  * Are types precise and minimal?
  * Any `any` or type assertions that could be avoided?

* **Immutability:**

  * Is state mutated directly anywhere?
  * Are readonly modifiers used appropriately?
  * Are immutable update patterns used?

* **Complexity:**

  * Could this be split into smaller functions/components?
  * Are there clever shortcuts that would confuse a new engineer?

* **React patterns:**

  * Are hooks used correctly?
  * Is component composition appropriate?
  * Are effects properly cleaned up?

* **Tests:**

  * Are there tests for new behavior?
  * Do tests describe behavior, not implementation?

### 12.2 Reviewer authority

* Reviewers push back on:

  * Unnecessary complexity.
  * Missing or weak types.
  * Mutable patterns where immutable would work.
  * Missing tests for non-trivial changes.

* "TypeScript allowed it" is not enough; humans guard readability.

---

## 13. Cultural Rules

* No unreviewed "quick hacks"; if it's worth merging, it's worth doing properly.
* Prefer "make it simple first" over "optimize prematurely".
* The future maintainer is probably not you. Write code that:

  * You could understand after six months away.
  * A new team member could navigate in a day.

If you are unsure whether a pattern is acceptable, **choose the simpler, more explicit option** or ask for guidance before baking in complexity.
