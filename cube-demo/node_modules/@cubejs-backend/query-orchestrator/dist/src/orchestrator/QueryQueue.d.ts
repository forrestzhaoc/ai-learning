export class QueryQueue {
    /**
     * Class constructor.
     *
     * @param {*} redisQueuePrefix
     * @param {*} options
     */
    constructor(redisQueuePrefix: any, options: any);
    /**
     * @type {string}
     */
    redisQueuePrefix: string;
    /**
     * @type {number}
     */
    concurrency: number;
    /**
     * @protected
     * @type {number}
     */
    protected continueWaitTimeout: number;
    /**
     * @protected
     * @type {number}
     */
    protected executionTimeout: number;
    /**
     * @protected
     * @type {number}
     */
    protected orphanedTimeout: number;
    /**
     * @protected
     * @type {number}
     */
    protected heartBeatInterval: number;
    /**
     * @protected
     * @type {function(QueryKeyHash, QueueId | null): Promise<void>}
     */
    protected sendProcessMessageFn: (arg0: QueryKeyHash, arg1: QueueId | null) => Promise<void>;
    /**
     * @protected
     * @param {QueryDef} query
     * @param {QueueId | null} queueId
     * @type {function(*): Promise<void>}
     */
    protected sendCancelMessageFn: (arg0: any) => Promise<void>;
    /**
     * @protected
     * @type {*}
     */
    protected queryHandlers: any;
    /**
     * @protected
     * @type {*}
     */
    protected cancelHandlers: any;
    /**
     * @protected
     * @type {function(string, *): void}
     */
    protected logger: (arg0: string, arg1: any) => void;
    processUid: any;
    /**
     * @type {QueueDriverInterface}
     */
    queueDriver: QueueDriverInterface;
    /**
     * @protected
     * @type {boolean}
     */
    protected skipQueue: boolean;
    /**
     * Persistent queries streams maps.
     */
    streams: Map<any, any>;
    /**
     * Notify streaming queries when streaming has been started and stream is available.
     */
    streamEvents: EventEmitter<[never]>;
    /**
     * Returns stream object which will be used to pipe data from data source.
     *
     * @param {QueryKeyHash} queryKeyHash
     * @return {QueryStream | undefined}
     */
    getQueryStream(queryKeyHash: QueryKeyHash): QueryStream | undefined;
    /**
     * @param {QueryKeyHash} key
     * @param {{ [alias: string]: string }} aliasNameToMember
     * @return {QueryStream}
     */
    createQueryStream(key: QueryKeyHash, aliasNameToMember: {
        [alias: string]: string;
    }): QueryStream;
    counter: number;
    generateQueueId(): number;
    /**
     * Push query to the queue and call `QueryQueue.reconcileQueue()` method if
     * `options.skipQueue` is set to `false`, execute query skipping queue
     * otherwise.
     *
     * @param {string} queryHandler
     * @param {*} queryKey
     * @param {*} query
     * @param {number=} priority
     * @param {*=} options
     * @returns {*}
     *
     * @throw {ContinueWaitError}
     */
    executeInQueue(queryHandler: string, queryKey: any, query: any, priority?: number | undefined, options?: any | undefined): any;
    /**
     * Parse query result.
     *
     * @param {*} result
     * @returns {*}
     *
     * @throw {Error}
     */
    parseResult(result: any): any;
    /**
     * Run query queue reconciliation flow by calling internal `reconcileQueueImpl`
     * method. Returns promise which will be resolved with the reconciliation
     * result.
     *
     * @returns {Promise}
     */
    reconcileQueue(): Promise<any>;
    reconcileAgain: boolean;
    reconcilePromise: any;
    shutdown(): Promise<boolean>;
    /**
     * Returns a full list of queued queries, including stalled, orphaned, active
     * and planned to be processed with their statuses and queries definitions.
     *
     * @returns {Promise<Object>}
     */
    getQueries(): Promise<any>;
    /**
     * Cancel query by its `queryKey`.
     *
     * @param {QueryKeyHash} queryKey
     * @param {QueueId | null} queueId
     * @returns {void}
     */
    cancelQuery(queryKey: QueryKeyHash, queueId: QueueId | null): void;
    /**
     * Reconciliation logic: cancel stalled and orphaned queries from the queue
     * and pick some planned to be processed queries to process.
     *
     * @private
     * @returns {Promise<void>}
     */
    private reconcileQueueImpl;
    /**
     * Apply query timeout to the query. Throw if query execution time takes more
     * than specified timeout. Returns resolved `promise` value.
     *
     * @param {Promise<*>} promise
     * @returns {Promise<*>}
     *
     * @throw
     */
    queryTimeout(promise: Promise<any>): Promise<any>;
    /**
     * Returns the list of queries planned to be processed and the list of active
     * queries.
     *
     * @returns {Array}
     */
    fetchQueryStageState(): any[];
    /**
     * Returns current state of the specified by the `stageQueryKey` query if it
     * exists.
     *
     * @param {*} stageQueryKey
     * @param {number=} priorityFilter
     * @param {Array=} queryStageState
     * @returns {Promise<undefined> | Promise<{ stage: string, timeElapsed: number }>}
     */
    getQueryStage(stageQueryKey: any, priorityFilter?: number | undefined, queryStageState?: any[] | undefined): Promise<undefined> | Promise<{
        stage: string;
        timeElapsed: number;
    }>;
    /**
     * Execute query without adding it to the queue.
     *
     * @param {*} query
     * @param {QueueId} queueId
     * @returns {Promise<{ result: undefined | Object, error: string | undefined }>}
     */
    processQuerySkipQueue(query: any, queueId: QueueId): Promise<{
        result: undefined | any;
        error: string | undefined;
    }>;
    /**
     * Processing query specified by the `queryKey`. This method encapsulate most
     * of the logic related with the queues updates, heartbeat, etc.
     *
     * @param {QueryKeyHash} queryKeyHashed
     * @param {QueueId | null} queueId Supported by new Cube Store and Memory
     * @return {Promise<{ result: undefined | Object, error: string | undefined }>}
     */
    processQuery(queryKeyHashed: QueryKeyHash, queueId: QueueId | null): Promise<{
        result: undefined | any;
        error: string | undefined;
    }>;
    /**
     * Processing cancel query flow.
     *
     * @param {QueryDef} query
     * @param {QueueId | null} queueId
     */
    processCancel(query: QueryDef, queueId: QueueId | null): Promise<void>;
    /**
     * Returns hash sum of the specified `queryKey`.
     *
     * @param {QueryKey} queryKey
     * @returns {QueryKeyHash}
     */
    redisHash(queryKey: QueryKey): QueryKeyHash;
}
import { QueryKeyHash } from '@cubejs-backend/base-driver';
import { QueueId } from '@cubejs-backend/base-driver';
import { QueueDriverInterface } from '@cubejs-backend/base-driver';
import { EventEmitter } from 'events';
import { QueryStream } from './QueryStream';
import { QueryDef } from '@cubejs-backend/base-driver';
import { QueryKey } from '@cubejs-backend/base-driver';
//# sourceMappingURL=QueryQueue.d.ts.map