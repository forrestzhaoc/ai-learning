/**
 * @implements {QueueDriverConnectionInterface}
 */
export class LocalQueueDriverConnection implements QueueDriverConnectionInterface {
    constructor(driver: any, options: any);
    redisQueuePrefix: any;
    continueWaitTimeout: any;
    heartBeatTimeout: any;
    concurrency: any;
    driver: any;
    results: any;
    resultPromises: any;
    queryDef: any;
    toProcess: any;
    recent: any;
    active: any;
    heartBeat: any;
    processingCounter: any;
    processingLocks: any;
    getQueueEventsBus: any;
    getQueriesToCancel(): Promise<QueryKeysTuple[]>;
    /**
     * @returns {Promise<GetActiveAndToProcessResponse>}
     */
    getActiveAndToProcess(): Promise<GetActiveAndToProcessResponse>;
    getResultPromise(resultListKey: any): any;
    getResultBlocking(queryKeyHash: any): Promise<any>;
    /**
     * Returns promise wich will be resolved with the specified by the
     * queryKey query result or null if query was not added to the
     * processing.
     * @param {*} queryKey
     * @returns {Promise<null | *>}
     */
    getResult(queryKey: any): Promise<null | any>;
    /**
     * @protected
     */
    protected queueArray(queueObj: any, orderFilterLessThan: any): any[];
    /**
     * @protected
     * @param queueObj
     * @param orderFilterLessThan
     * @returns {QueryKeysTuple[]}
     */
    protected queueArrayAsTuple(queueObj: any, orderFilterLessThan: any): QueryKeysTuple[];
    /**
     * Adds specified by the queryKey query to the queue, returns tuple
     * with the operation result.
     *
     * @typedef {[added: number, _b: null, _c: null, toProcessLength: number, addedTime: number]} AddedTuple
     *
     * @param {number} keyScore
     * @param {*} queryKey
     * @param {number} orphanedTime
     * @param {string} queryHandler (for the regular query is eq to 'query')
     * @param {*} query
     * @param {number} priority
     * @param {*} options
     * @returns {AddedTuple}
     */
    addToQueue(keyScore: number, queryKey: any, orphanedTime: number, queryHandler: string, query: any, priority: number, options: any): [added: number, _b: null, _c: null, toProcessLength: number, addedTime: number];
    getToProcessQueries(): QueryKeysTuple[];
    getActiveQueries(): QueryKeysTuple[];
    getQueryAndRemove(queryKey: any): Promise<any[]>;
    cancelQuery(queryKey: any): Promise<any>;
    setResultAndRemoveQuery(queryKey: any, executionResult: any, processingId: any): Promise<boolean>;
    getNextProcessingId(): any;
    getOrphanedQueries(): QueryKeysTuple[];
    getStalledQueries(): QueryKeysTuple[];
    getQueryStageState(onlyKeys: any): Promise<any[]>;
    getQueryDef(queryKey: any): Promise<any>;
    /**
     * Updates heart beat for the processing query by its `queryKey`.
     *
     * @param {string} queryKey
     */
    updateHeartBeat(queryKey: string): void;
    retrieveForProcessing(queryKey: any, processingId: any): any[];
    freeProcessingLock(queryKey: any, processingId: any, activated: any): void;
    optimisticQueryUpdate(queryKey: any, toUpdate: any, processingId: any): Promise<boolean>;
    release(): void;
    /**
     * Returns cache key to the specified by the queryKey query and the
     * specified by the suffix query state.
     * @param {*} queryKey
     * @param {string} suffix
     * @returns {string}
     */
    queryRedisKey(queryKey: any, suffix: string): string;
    /**
     * Returns cache key to the cached query result.
     * @param {*} queryKey
     * @returns {string}
     */
    resultListKey(queryKey: any): string;
    /**
     * Returns hash sum of the query specified by the queryKey.
     * @param {*} queryKey
     * @returns {string}
     */
    redisHash(queryKey: any): string;
}
/**
 * @implements {QueueDriverInterface}
 */
export class LocalQueueDriver extends BaseQueueDriver implements QueueDriverInterface {
    constructor(options: any);
    options: any;
    results: any;
    resultPromises: any;
    queryDef: any;
    toProcess: any;
    recent: any;
    active: any;
    heartBeat: any;
    processingCounter: any;
    processingLocks: any;
    createConnection(): LocalQueueDriverConnection;
    release(client: any): void;
}
import { QueueDriverConnectionInterface } from '@cubejs-backend/base-driver';
import { QueueDriverInterface } from '@cubejs-backend/base-driver';
import { BaseQueueDriver } from './BaseQueueDriver';
//# sourceMappingURL=LocalQueueDriver.d.ts.map