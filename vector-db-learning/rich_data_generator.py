"""
丰富的数据生成器
创建包含多种场景、属性、关系的复杂数据集
用于展示向量数据库的优势
"""

from models.graph_models import Node, Relationship, GraphData
import random

def create_rich_data():
    """创建丰富的数据集，包含多种场景"""
    graph_data = GraphData()
    
    # ========== 人物节点 ==========
    # 动作片演员
    action_actors = [
        Node("p1", "Person", {
            "name": "Tom Hanks",
            "born": 1956,
            "nationality": "American",
            "profession": "Actor",
            "genre": "Drama, Comedy",
            "awards": "Oscar Winner",
            "bio": "Academy Award-winning actor known for versatile roles"
        }),
        Node("p2", "Person", {
            "name": "Keanu Reeves",
            "born": 1964,
            "nationality": "Canadian",
            "profession": "Actor",
            "genre": "Action, Sci-Fi",
            "awards": "MTV Movie Award",
            "bio": "Action star known for The Matrix and John Wick series"
        }),
        Node("p3", "Person", {
            "name": "Carrie-Anne Moss",
            "born": 1967,
            "nationality": "Canadian",
            "profession": "Actress",
            "genre": "Action, Sci-Fi",
            "bio": "Known for her role as Trinity in The Matrix"
        }),
        Node("p4", "Person", {
            "name": "Laurence Fishburne",
            "born": 1961,
            "nationality": "American",
            "profession": "Actor",
            "genre": "Drama, Action",
            "awards": "Tony Award",
            "bio": "Acclaimed actor with diverse filmography"
        }),
        Node("p5", "Person", {
            "name": "Hugo Weaving",
            "born": 1960,
            "nationality": "Australian",
            "profession": "Actor",
            "genre": "Action, Fantasy",
            "bio": "Versatile actor known for Matrix and Lord of the Rings"
        }),
        Node("p6", "Person", {
            "name": "Lana Wachowski",
            "born": 1965,
            "nationality": "American",
            "profession": "Director, Writer",
            "genre": "Sci-Fi, Action",
            "awards": "Saturn Award",
            "bio": "Visionary filmmaker and co-creator of The Matrix"
        }),
        Node("p7", "Person", {
            "name": "Lilly Wachowski",
            "born": 1967,
            "nationality": "American",
            "profession": "Director, Writer",
            "genre": "Sci-Fi, Action",
            "awards": "Saturn Award",
            "bio": "Visionary filmmaker and co-creator of The Matrix"
        }),
        Node("p8", "Person", {
            "name": "Robert Zemeckis",
            "born": 1952,
            "nationality": "American",
            "profession": "Director",
            "genre": "Drama, Fantasy",
            "awards": "Oscar Winner",
            "bio": "Academy Award-winning director"
        }),
        Node("p9", "Person", {
            "name": "Robin Wright",
            "born": 1966,
            "nationality": "American",
            "profession": "Actress",
            "genre": "Drama",
            "bio": "Acclaimed actress known for dramatic roles"
        }),
        Node("p10", "Person", {
            "name": "Gary Sinise",
            "born": 1955,
            "nationality": "American",
            "profession": "Actor",
            "genre": "Drama",
            "bio": "Versatile character actor"
        }),
    ]
    
    # 添加更多演员
    more_actors = [
        Node("p11", "Person", {
            "name": "Leonardo DiCaprio",
            "born": 1974,
            "nationality": "American",
            "profession": "Actor",
            "genre": "Drama, Thriller",
            "awards": "Oscar Winner",
            "bio": "Academy Award winner known for intense dramatic performances"
        }),
        Node("p12", "Person", {
            "name": "Kate Winslet",
            "born": 1975,
            "nationality": "British",
            "profession": "Actress",
            "genre": "Drama, Romance",
            "awards": "Oscar Winner",
            "bio": "Academy Award-winning British actress"
        }),
        Node("p13", "Person", {
            "name": "Christopher Nolan",
            "born": 1970,
            "nationality": "British",
            "profession": "Director, Writer",
            "genre": "Sci-Fi, Thriller",
            "awards": "Oscar Nominee",
            "bio": "Visionary director known for complex narratives"
        }),
        Node("p14", "Person", {
            "name": "Christian Bale",
            "born": 1974,
            "nationality": "British",
            "profession": "Actor",
            "genre": "Action, Drama",
            "awards": "Oscar Winner",
            "bio": "Method actor known for physical transformations"
        }),
        Node("p15", "Person", {
            "name": "Heath Ledger",
            "born": 1979,
            "nationality": "Australian",
            "profession": "Actor",
            "genre": "Drama, Action",
            "awards": "Oscar Winner",
            "bio": "Academy Award winner, known for The Dark Knight"
        }),
        Node("p16", "Person", {
            "name": "Meryl Streep",
            "born": 1949,
            "nationality": "American",
            "profession": "Actress",
            "genre": "Drama, Comedy",
            "awards": "Oscar Winner (3 times)",
            "bio": "Most nominated actor in Oscar history"
        }),
        Node("p17", "Person", {
            "name": "Steven Spielberg",
            "born": 1946,
            "nationality": "American",
            "profession": "Director, Producer",
            "genre": "Adventure, Drama, Sci-Fi",
            "awards": "Oscar Winner (3 times)",
            "bio": "Legendary filmmaker and producer"
        }),
        Node("p18", "Person", {
            "name": "Tom Cruise",
            "born": 1962,
            "nationality": "American",
            "profession": "Actor, Producer",
            "genre": "Action, Thriller",
            "awards": "Golden Globe Winner",
            "bio": "Action star known for performing own stunts"
        }),
        Node("p19", "Person", {
            "name": "Quentin Tarantino",
            "born": 1963,
            "nationality": "American",
            "profession": "Director, Writer",
            "genre": "Crime, Drama",
            "awards": "Oscar Winner",
            "bio": "Auteur filmmaker known for nonlinear storytelling"
        }),
        Node("p20", "Person", {
            "name": "Samuel L. Jackson",
            "born": 1948,
            "nationality": "American",
            "profession": "Actor",
            "genre": "Action, Crime, Drama",
            "awards": "Oscar Nominee",
            "bio": "Prolific actor with over 100 film credits"
        }),
    ]
    
    all_people = action_actors + more_actors
    
    # ========== 电影节点 ==========
    movies = [
        # Matrix 系列
        Node("m1", "Movie", {
            "title": "The Matrix",
            "released": 1999,
            "genre": "Sci-Fi, Action",
            "tagline": "Welcome to the Real World",
            "rating": 8.7,
            "box_office": 467000000,
            "language": "English",
            "country": "USA, Australia",
            "runtime": 136,
            "plot": "A computer hacker learns about the true nature of reality"
        }),
        Node("m2", "Movie", {
            "title": "The Matrix Reloaded",
            "released": 2003,
            "genre": "Sci-Fi, Action",
            "tagline": "Free your mind",
            "rating": 7.2,
            "box_office": 742000000,
            "language": "English",
            "country": "USA, Australia",
            "runtime": 138,
            "plot": "Neo and the resistance continue their fight against the machines"
        }),
        Node("m3", "Movie", {
            "title": "The Matrix Revolutions",
            "released": 2003,
            "genre": "Sci-Fi, Action",
            "tagline": "Everything that has a beginning has an end",
            "rating": 6.8,
            "box_office": 427000000,
            "language": "English",
            "country": "USA, Australia",
            "runtime": 129,
            "plot": "The final battle between humans and machines"
        }),
        
        # 经典电影
        Node("m4", "Movie", {
            "title": "Forrest Gump",
            "released": 1994,
            "genre": "Drama, Romance",
            "tagline": "Life is like a box of chocolates",
            "rating": 8.8,
            "box_office": 678000000,
            "language": "English",
            "country": "USA",
            "runtime": 142,
            "plot": "The presidencies of Kennedy and Johnson through the eyes of an Alabama man"
        }),
        Node("m5", "Movie", {
            "title": "The Green Mile",
            "released": 1999,
            "genre": "Drama, Fantasy",
            "tagline": "Walk a mile you'll never forget",
            "rating": 8.6,
            "box_office": 286000000,
            "language": "English",
            "country": "USA",
            "runtime": 189,
            "plot": "The lives of guards on Death Row are affected by one of their charges"
        }),
        
        # 更多电影
        Node("m6", "Movie", {
            "title": "Titanic",
            "released": 1997,
            "genre": "Romance, Drama",
            "tagline": "Nothing on Earth could come between them",
            "rating": 7.9,
            "box_office": 2200000000,
            "language": "English",
            "country": "USA",
            "runtime": 194,
            "plot": "A seventeen-year-old aristocrat falls in love with a kind but poor artist"
        }),
        Node("m7", "Movie", {
            "title": "Inception",
            "released": 2010,
            "genre": "Sci-Fi, Thriller",
            "tagline": "Your mind is the scene of the crime",
            "rating": 8.8,
            "box_office": 836000000,
            "language": "English",
            "country": "USA, UK",
            "runtime": 148,
            "plot": "A thief who steals corporate secrets through dream-sharing technology"
        }),
        Node("m8", "Movie", {
            "title": "The Dark Knight",
            "released": 2008,
            "genre": "Action, Crime, Drama",
            "tagline": "Why so serious?",
            "rating": 9.0,
            "box_office": 1005000000,
            "language": "English",
            "country": "USA, UK",
            "runtime": 152,
            "plot": "Batman faces the Joker in a battle for Gotham's soul"
        }),
        Node("m9", "Movie", {
            "title": "The Prestige",
            "released": 2006,
            "genre": "Drama, Mystery, Thriller",
            "tagline": "Are you watching closely?",
            "rating": 8.5,
            "box_office": 109000000,
            "language": "English",
            "country": "USA, UK",
            "runtime": 130,
            "plot": "Two stage magicians engage in competitive one-upmanship"
        }),
        Node("m10", "Movie", {
            "title": "Interstellar",
            "released": 2014,
            "genre": "Sci-Fi, Drama",
            "tagline": "Mankind was born on Earth. It was never meant to die here.",
            "rating": 8.6,
            "box_office": 677000000,
            "language": "English",
            "country": "USA, UK",
            "runtime": 169,
            "plot": "A team of explorers travel through a wormhole in space"
        }),
        Node("m11", "Movie", {
            "title": "Mission: Impossible",
            "released": 1996,
            "genre": "Action, Thriller",
            "tagline": "Expect the Impossible",
            "rating": 7.1,
            "box_office": 457000000,
            "language": "English",
            "country": "USA",
            "runtime": 110,
            "plot": "An American agent, under false suspicion of disloyalty, must discover and expose the real spy"
        }),
        Node("m12", "Movie", {
            "title": "Pulp Fiction",
            "released": 1994,
            "genre": "Crime, Drama",
            "tagline": "Just because you are a character doesn't mean you have character",
            "rating": 8.9,
            "box_office": 214000000,
            "language": "English",
            "country": "USA",
            "runtime": 154,
            "plot": "The lives of two mob hitmen, a boxer, and others intertwine"
        }),
        Node("m13", "Movie", {
            "title": "The Devil Wears Prada",
            "released": 2006,
            "genre": "Comedy, Drama",
            "tagline": "Meet Andy Sachs. A million girls would kill to have her job. She's not one of them.",
            "rating": 6.9,
            "box_office": 326000000,
            "language": "English",
            "country": "USA",
            "runtime": 109,
            "plot": "A smart but sensible new graduate lands a job as an assistant to a demanding fashion magazine editor"
        }),
        Node("m14", "Movie", {
            "title": "Schindler's List",
            "released": 1993,
            "genre": "Drama, History",
            "tagline": "Whoever saves one life, saves the world entire",
            "rating": 9.0,
            "box_office": 321000000,
            "language": "English, German, Hebrew",
            "country": "USA",
            "runtime": 195,
            "plot": "In German-occupied Poland, Oskar Schindler gradually becomes concerned for his Jewish workforce"
        }),
        Node("m15", "Movie", {
            "title": "Saving Private Ryan",
            "released": 1998,
            "genre": "Drama, War",
            "tagline": "The mission is a man",
            "rating": 8.6,
            "box_office": 482000000,
            "language": "English, French, German",
            "country": "USA",
            "runtime": 169,
            "plot": "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines"
        }),
    ]
    
    # 添加所有节点
    for person in all_people:
        graph_data.add_node(person)
    for movie in movies:
        graph_data.add_node(movie)
    
    # ========== 关系 ==========
    relationships = [
        # Matrix 系列关系
        Relationship("r1", "p2", "m1", "ACTED_IN", {"roles": ["Neo"], "billing": 1}),
        Relationship("r2", "p3", "m1", "ACTED_IN", {"roles": ["Trinity"], "billing": 2}),
        Relationship("r3", "p4", "m1", "ACTED_IN", {"roles": ["Morpheus"], "billing": 3}),
        Relationship("r4", "p5", "m1", "ACTED_IN", {"roles": ["Agent Smith"], "billing": 4}),
        Relationship("r5", "p6", "m1", "DIRECTED", {}),
        Relationship("r6", "p7", "m1", "DIRECTED", {}),
        Relationship("r7", "p6", "m1", "WROTE", {}),
        Relationship("r7b", "p7", "m1", "WROTE", {}),
        
        Relationship("r8", "p2", "m2", "ACTED_IN", {"roles": ["Neo"], "billing": 1}),
        Relationship("r9", "p3", "m2", "ACTED_IN", {"roles": ["Trinity"], "billing": 2}),
        Relationship("r10", "p4", "m2", "ACTED_IN", {"roles": ["Morpheus"], "billing": 3}),
        Relationship("r11", "p6", "m2", "DIRECTED", {}),
        Relationship("r12", "p7", "m2", "DIRECTED", {}),
        
        Relationship("r13", "p2", "m3", "ACTED_IN", {"roles": ["Neo"], "billing": 1}),
        Relationship("r14", "p3", "m3", "ACTED_IN", {"roles": ["Trinity"], "billing": 2}),
        Relationship("r15", "p4", "m3", "ACTED_IN", {"roles": ["Morpheus"], "billing": 3}),
        Relationship("r16", "p6", "m3", "DIRECTED", {}),
        Relationship("r17", "p7", "m3", "DIRECTED", {}),
        
        # Forrest Gump
        Relationship("r18", "p1", "m4", "ACTED_IN", {"roles": ["Forrest Gump"], "billing": 1}),
        Relationship("r19", "p9", "m4", "ACTED_IN", {"roles": ["Jenny Curran"], "billing": 2}),
        Relationship("r20", "p10", "m4", "ACTED_IN", {"roles": ["Lieutenant Dan Taylor"], "billing": 3}),
        Relationship("r21", "p8", "m4", "DIRECTED", {}),
        
        # The Green Mile
        Relationship("r22", "p1", "m5", "ACTED_IN", {"roles": ["Paul Edgecomb"], "billing": 1}),
        
        # Titanic
        Relationship("r23", "p11", "m6", "ACTED_IN", {"roles": ["Jack Dawson"], "billing": 1}),
        Relationship("r24", "p12", "m6", "ACTED_IN", {"roles": ["Rose DeWitt Bukater"], "billing": 2}),
        Relationship("r25", "p17", "m6", "DIRECTED", {}),
        Relationship("r26", "p17", "m6", "PRODUCED", {}),
        
        # Inception
        Relationship("r27", "p13", "m7", "DIRECTED", {}),
        Relationship("r28", "p13", "m7", "WROTE", {}),
        Relationship("r29", "p14", "m7", "ACTED_IN", {"roles": ["Dom Cobb"], "billing": 1}),
        
        # The Dark Knight
        Relationship("r30", "p13", "m8", "DIRECTED", {}),
        Relationship("r31", "p13", "m8", "WROTE", {}),
        Relationship("r32", "p14", "m8", "ACTED_IN", {"roles": ["Bruce Wayne / Batman"], "billing": 1}),
        Relationship("r33", "p15", "m8", "ACTED_IN", {"roles": ["Joker"], "billing": 2}),
        
        # The Prestige
        Relationship("r34", "p13", "m9", "DIRECTED", {}),
        Relationship("r35", "p14", "m9", "ACTED_IN", {"roles": ["Alfred Borden"], "billing": 1}),
        Relationship("r36", "p15", "m9", "ACTED_IN", {"roles": ["Alfred Borden"], "billing": 1}),
        
        # Interstellar
        Relationship("r37", "p13", "m10", "DIRECTED", {}),
        Relationship("r38", "p13", "m10", "WROTE", {}),
        Relationship("r39", "p14", "m10", "ACTED_IN", {"roles": ["Cooper"], "billing": 1}),
        
        # Mission: Impossible
        Relationship("r40", "p18", "m11", "ACTED_IN", {"roles": ["Ethan Hunt"], "billing": 1}),
        Relationship("r41", "p18", "m11", "PRODUCED", {}),
        
        # Pulp Fiction
        Relationship("r42", "p19", "m12", "DIRECTED", {}),
        Relationship("r43", "p19", "m12", "WROTE", {}),
        Relationship("r44", "p20", "m12", "ACTED_IN", {"roles": ["Jules Winnfield"], "billing": 2}),
        
        # The Devil Wears Prada
        Relationship("r45", "p12", "m13", "ACTED_IN", {"roles": ["Andy Sachs"], "billing": 1}),
        Relationship("r46", "p16", "m13", "ACTED_IN", {"roles": ["Miranda Priestly"], "billing": 2}),
        
        # Schindler's List
        Relationship("r47", "p17", "m14", "DIRECTED", {}),
        Relationship("r48", "p17", "m14", "PRODUCED", {}),
        
        # Saving Private Ryan
        Relationship("r49", "p17", "m15", "DIRECTED", {}),
        Relationship("r50", "p1", "m15", "ACTED_IN", {"roles": ["Captain John H. Miller"], "billing": 1}),
        
        # 评论关系
        Relationship("r51", "p1", "m7", "REVIEWED", {"rating": 9, "comment": "Mind-bending masterpiece"}),
        Relationship("r52", "p2", "m8", "REVIEWED", {"rating": 10, "comment": "Best superhero movie ever"}),
        Relationship("r53", "p11", "m1", "REVIEWED", {"rating": 8, "comment": "Revolutionary sci-fi"}),
    ]
    
    # 添加所有关系
    for rel in relationships:
        source_node = graph_data.get_node(rel.source_id)
        target_node = graph_data.get_node(rel.target_id)
        if source_node and target_node:
            graph_data.add_relationship(rel)
    
    return graph_data


