import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => cleanup());

// React Router builds a fetch `Request` for every navigation and hands it an AbortSignal. Under
// jsdom that signal is jsdom's, which Node's undici `Request` rejects. Nothing here aborts a
// navigation, so tests drop the signal rather than swap the whole environment.
const NativeRequest = globalThis.Request;
class TestRequest extends NativeRequest {
  constructor(input: RequestInfo | URL, init?: RequestInit) {
    const { signal: _signal, ...rest } = init ?? {};
    super(input, rest);
  }
}
globalThis.Request = TestRequest as typeof Request;
