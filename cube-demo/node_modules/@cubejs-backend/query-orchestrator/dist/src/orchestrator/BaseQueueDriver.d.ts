import { QueryKey, QueryKeyHash, QueueDriverConnectionInterface, QueueDriverInterface } from '@cubejs-backend/base-driver';
export declare abstract class BaseQueueDriver implements QueueDriverInterface {
    protected processUid: string;
    constructor(processUid: string);
    redisHash(queryKey: QueryKey): QueryKeyHash;
    abstract createConnection(): Promise<QueueDriverConnectionInterface>;
    abstract release(connection: QueueDriverConnectionInterface): void;
}
//# sourceMappingURL=BaseQueueDriver.d.ts.map