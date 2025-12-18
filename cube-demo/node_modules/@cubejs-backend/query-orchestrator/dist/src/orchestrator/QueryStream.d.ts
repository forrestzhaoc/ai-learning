/// <reference types="node" />
import * as stream from 'stream';
export declare class QueryStream extends stream.Transform {
    private timeout;
    private timer;
    queryKey: string;
    streams: Map<string, stream.Stream>;
    aliasNameToMember: {
        [alias: string]: string;
    };
    counter: number;
    /**
     * @constructor
     */
    constructor({ key, streams, aliasNameToMember, }: {
        key: string;
        streams: Map<string, stream.Stream>;
        aliasNameToMember: {
            [alias: string]: string;
        } | null;
    });
    /**
     * @override
     */
    _transform(chunk: any, encoding: any, callback: any): void;
    /**
     * @override
     */
    _destroy(error: any, callback: any): void;
    /**
     * Reset destroyer timeout.
     */
    debounce(): void;
}
//# sourceMappingURL=QueryStream.d.ts.map