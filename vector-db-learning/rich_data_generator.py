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
        
        # 中国演员和导演
        Node("p21", "Person", {
            "name": "Zhang Yimou",
            "born": 1951,
            "nationality": "Chinese",
            "profession": "Director",
            "genre": "Drama, Action, Historical",
            "awards": "Golden Bear, Golden Lion",
            "bio": "Acclaimed Chinese filmmaker known for visually stunning films"
        }),
        Node("p22", "Person", {
            "name": "Ang Lee",
            "born": 1954,
            "nationality": "Taiwanese",
            "profession": "Director",
            "genre": "Drama, Romance, Action",
            "awards": "Oscar Winner (2 times)",
            "bio": "Academy Award-winning director known for diverse filmography"
        }),
        Node("p23", "Person", {
            "name": "Jet Li",
            "born": 1963,
            "nationality": "Chinese",
            "profession": "Actor",
            "genre": "Action, Martial Arts",
            "bio": "Martial arts actor and former wushu champion"
        }),
        Node("p24", "Person", {
            "name": "Zhang Ziyi",
            "born": 1979,
            "nationality": "Chinese",
            "profession": "Actress",
            "genre": "Action, Drama",
            "bio": "Acclaimed Chinese actress known for martial arts films"
        }),
        Node("p25", "Person", {
            "name": "Donnie Yen",
            "born": 1963,
            "nationality": "Chinese",
            "profession": "Actor",
            "genre": "Action, Martial Arts",
            "bio": "Martial arts actor and action choreographer"
        }),
        Node("p26", "Person", {
            "name": "Wu Jing",
            "born": 1974,
            "nationality": "Chinese",
            "profession": "Actor, Director",
            "genre": "Action, Military",
            "bio": "Chinese action star and director"
        }),
        Node("p27", "Person", {
            "name": "Frant Gwo",
            "born": 1980,
            "nationality": "Chinese",
            "profession": "Director, Writer",
            "genre": "Sci-Fi, Action",
            "bio": "Chinese filmmaker known for sci-fi blockbusters"
        }),
        
        # 更多中国演员和导演
        Node("p28", "Person", {
            "name": "Gong Li",
            "born": 1965,
            "nationality": "Chinese",
            "profession": "Actress",
            "genre": "Drama",
            "awards": "Golden Rooster Award",
            "bio": "Acclaimed Chinese actress, frequent collaborator with Zhang Yimou"
        }),
        Node("p29", "Person", {
            "name": "Tony Leung",
            "born": 1962,
            "nationality": "Hong Kong",
            "profession": "Actor",
            "genre": "Drama, Action",
            "awards": "Cannes Best Actor",
            "bio": "Acclaimed Hong Kong actor"
        }),
        Node("p30", "Person", {
            "name": "Maggie Cheung",
            "born": 1964,
            "nationality": "Hong Kong",
            "profession": "Actress",
            "genre": "Drama, Action",
            "awards": "Cannes Best Actress",
            "bio": "Acclaimed Hong Kong actress"
        }),
        Node("p31", "Person", {
            "name": "Chow Yun-fat",
            "born": 1955,
            "nationality": "Hong Kong",
            "profession": "Actor",
            "genre": "Action, Drama",
            "bio": "Legendary Hong Kong actor"
        }),
        Node("p32", "Person", {
            "name": "Jackie Chan",
            "born": 1954,
            "nationality": "Hong Kong",
            "profession": "Actor, Director, Stunt Performer",
            "genre": "Action, Comedy",
            "awards": "Oscar Honorary Award",
            "bio": "International action star and martial artist"
        }),
        Node("p33", "Person", {
            "name": "Stephen Chow",
            "born": 1962,
            "nationality": "Hong Kong",
            "profession": "Actor, Director, Writer",
            "genre": "Comedy, Action",
            "bio": "Comedy filmmaker and actor"
        }),
        Node("p34", "Person", {
            "name": "Wong Kar-wai",
            "born": 1958,
            "nationality": "Hong Kong",
            "profession": "Director, Writer",
            "genre": "Drama, Romance",
            "awards": "Cannes Best Director",
            "bio": "Acclaimed Hong Kong filmmaker"
        }),
        Node("p35", "Person", {
            "name": "Chen Kaige",
            "born": 1952,
            "nationality": "Chinese",
            "profession": "Director",
            "genre": "Drama, Historical",
            "awards": "Palme d'Or",
            "bio": "Fifth Generation Chinese filmmaker"
        }),
        Node("p36", "Person", {
            "name": "Jia Zhangke",
            "born": 1970,
            "nationality": "Chinese",
            "profession": "Director, Writer",
            "genre": "Drama, Documentary",
            "awards": "Golden Lion",
            "bio": "Sixth Generation Chinese filmmaker"
        }),
        Node("p37", "Person", {
            "name": "Wang Baoqiang",
            "born": 1984,
            "nationality": "Chinese",
            "profession": "Actor, Director",
            "genre": "Comedy, Drama",
            "bio": "Chinese actor and director"
        }),
        Node("p38", "Person", {
            "name": "Huang Bo",
            "born": 1974,
            "nationality": "Chinese",
            "profession": "Actor, Director",
            "genre": "Comedy, Drama",
            "bio": "Chinese actor and filmmaker"
        }),
        Node("p39", "Person", {
            "name": "Shen Teng",
            "born": 1979,
            "nationality": "Chinese",
            "profession": "Actor, Comedian",
            "genre": "Comedy",
            "bio": "Chinese comedian and actor"
        }),
        Node("p40", "Person", {
            "name": "Zhou Dongyu",
            "born": 1992,
            "nationality": "Chinese",
            "profession": "Actress",
            "genre": "Drama",
            "awards": "Golden Horse Award",
            "bio": "Chinese actress"
        }),
        Node("p41", "Person", {
            "name": "Yi Yangqianxi",
            "born": 2000,
            "nationality": "Chinese",
            "profession": "Actor, Singer",
            "genre": "Drama, Action",
            "bio": "Chinese actor and pop star"
        }),
        Node("p42", "Person", {
            "name": "Deng Chao",
            "born": 1979,
            "nationality": "Chinese",
            "profession": "Actor, Director",
            "genre": "Comedy, Drama",
            "bio": "Chinese actor and director"
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
        
        # 中国电影
        Node("m16", "Movie", {
            "title": "Crouching Tiger, Hidden Dragon",
            "released": 2000,
            "genre": "Action, Drama, Romance",
            "tagline": "A timeless story of strength, secrets and two warriors who would never surrender",
            "rating": 7.9,
            "box_office": 213000000,
            "language": "Mandarin, English",
            "country": "China, Taiwan, USA, Hong Kong",
            "runtime": 120,
            "plot": "A young Chinese warrior steals a sword from a famed swordsman and then escapes into a world of romantic adventure"
        }),
        Node("m17", "Movie", {
            "title": "Hero",
            "released": 2002,
            "genre": "Action, Drama, History",
            "tagline": "Before China was one, it was seven warring kingdoms",
            "rating": 7.9,
            "box_office": 177000000,
            "language": "Mandarin",
            "country": "China, Hong Kong",
            "runtime": 99,
            "plot": "A defense officer, Nameless, was summoned by the King of Qin regarding his success of terminating three warriors"
        }),
        Node("m18", "Movie", {
            "title": "House of Flying Daggers",
            "released": 2004,
            "genre": "Action, Drama, Romance",
            "tagline": "Love is a lie, only death is real",
            "rating": 7.5,
            "box_office": 93000000,
            "language": "Mandarin",
            "country": "China, Hong Kong",
            "runtime": 119,
            "plot": "A romantic police captain breaks a beautiful member of a revolutionary group out of prison to help her rejoin her fellows"
        }),
        Node("m19", "Movie", {
            "title": "The Wandering Earth",
            "released": 2019,
            "genre": "Sci-Fi, Action, Drama",
            "tagline": "For the survival of humanity",
            "rating": 6.4,
            "box_office": 700000000,
            "language": "Mandarin, English",
            "country": "China",
            "runtime": 125,
            "plot": "As the sun is dying out, people all around the world build giant planet thrusters to move Earth out of its orbit"
        }),
        Node("m20", "Movie", {
            "title": "Wolf Warrior 2",
            "released": 2017,
            "genre": "Action, Thriller",
            "tagline": "Whoever offends China will be hunted down no matter how far the target is",
            "rating": 6.0,
            "box_office": 870000000,
            "language": "Mandarin, English",
            "country": "China",
            "runtime": 126,
            "plot": "A Chinese special forces soldier fights against mercenaries in Africa"
        }),
        Node("m21", "Movie", {
            "title": "Ne Zha",
            "released": 2019,
            "genre": "Animation, Action, Adventure",
            "tagline": "I am the master of my own fate",
            "rating": 7.5,
            "box_office": 720000000,
            "language": "Mandarin",
            "country": "China",
            "runtime": 110,
            "plot": "A boy born with unique powers tries to find a place in the world"
        }),
        
        # 更多中国电影
        Node("m22", "Movie", {
            "title": "Farewell My Concubine",
            "released": 1993,
            "genre": "Drama, Music, Romance",
            "tagline": "Two men, one woman, and half a century of Chinese history",
            "rating": 8.1,
            "box_office": 5200000,
            "language": "Mandarin",
            "country": "China, Hong Kong",
            "runtime": 171,
            "plot": "Two boys meet at an opera training school in Peking in 1924"
        }),
        Node("m23", "Movie", {
            "title": "Raise the Red Lantern",
            "released": 1991,
            "genre": "Drama",
            "tagline": "Tradition is a trap",
            "rating": 8.1,
            "box_office": 2600000,
            "language": "Mandarin",
            "country": "China, Hong Kong, Taiwan",
            "runtime": 125,
            "plot": "A young woman becomes the fourth wife of a wealthy man"
        }),
        Node("m24", "Movie", {
            "title": "To Live",
            "released": 1994,
            "genre": "Drama, History",
            "tagline": "A family's journey through China's turbulent history",
            "rating": 8.3,
            "box_office": 2300000,
            "language": "Mandarin",
            "country": "China, Hong Kong",
            "runtime": 132,
            "plot": "A family struggles to survive through China's political upheavals"
        }),
        Node("m25", "Movie", {
            "title": "Red Sorghum",
            "released": 1987,
            "genre": "Drama, War",
            "tagline": "A story of love and resistance",
            "rating": 7.7,
            "box_office": 0,
            "language": "Mandarin",
            "country": "China",
            "runtime": 91,
            "plot": "A young woman is forced to marry an old winery owner"
        }),
        Node("m26", "Movie", {
            "title": "The Grandmaster",
            "released": 2013,
            "genre": "Action, Biography, Drama",
            "tagline": "The legend of Ip Man",
            "rating": 6.5,
            "box_office": 64000000,
            "language": "Mandarin, Cantonese",
            "country": "China, Hong Kong",
            "runtime": 108,
            "plot": "The story of martial-arts master Ip Man, the man who trained Bruce Lee"
        }),
        Node("m27", "Movie", {
            "title": "Ip Man",
            "released": 2008,
            "genre": "Action, Biography, Drama",
            "tagline": "The legend begins",
            "rating": 8.0,
            "box_office": 22000000,
            "language": "Cantonese, Mandarin",
            "country": "China, Hong Kong",
            "runtime": 106,
            "plot": "During the Japanese invasion of 1937, a wealthy martial artist is forced to leave his home"
        }),
        Node("m28", "Movie", {
            "title": "Ip Man 2",
            "released": 2010,
            "genre": "Action, Biography, Drama",
            "tagline": "The legend is born",
            "rating": 7.5,
            "box_office": 19000000,
            "language": "Cantonese, Mandarin",
            "country": "China, Hong Kong",
            "runtime": 108,
            "plot": "Ip Man moves to Hong Kong after the war to start a new life"
        }),
        Node("m29", "Movie", {
            "title": "Detective Chinatown",
            "released": 2015,
            "genre": "Action, Comedy, Mystery",
            "tagline": "A hilarious detective adventure",
            "rating": 6.5,
            "box_office": 125000000,
            "language": "Mandarin",
            "country": "China",
            "runtime": 135,
            "plot": "A detective and his nephew travel to Bangkok to solve a murder case"
        }),
        Node("m30", "Movie", {
            "title": "Detective Chinatown 2",
            "released": 2018,
            "genre": "Action, Comedy, Mystery",
            "tagline": "The adventure continues in New York",
            "rating": 6.1,
            "box_office": 544000000,
            "language": "Mandarin, English",
            "country": "China",
            "runtime": 121,
            "plot": "The detective duo travels to New York to solve a new case"
        }),
        Node("m31", "Movie", {
            "title": "Monster Hunt",
            "released": 2015,
            "genre": "Action, Adventure, Comedy",
            "tagline": "A fantasy adventure",
            "rating": 5.8,
            "box_office": 385000000,
            "language": "Mandarin",
            "country": "China",
            "runtime": 118,
            "plot": "A young man discovers he is pregnant with a monster prince"
        }),
        Node("m32", "Movie", {
            "title": "The Mermaid",
            "released": 2016,
            "genre": "Comedy, Fantasy, Romance",
            "tagline": "A love story between a mermaid and a businessman",
            "rating": 6.2,
            "box_office": 553000000,
            "language": "Mandarin, Cantonese",
            "country": "China",
            "runtime": 94,
            "plot": "A mermaid is sent to assassinate a businessman but falls in love"
        }),
        Node("m33", "Movie", {
            "title": "Operation Red Sea",
            "released": 2018,
            "genre": "Action, Drama, Thriller",
            "tagline": "A Chinese naval special forces operation",
            "rating": 6.7,
            "box_office": 579000000,
            "language": "Mandarin, Arabic, English",
            "country": "China",
            "runtime": 142,
            "plot": "Chinese naval special forces rescue Chinese citizens from a war-torn country"
        }),
        Node("m34", "Movie", {
            "title": "Dying to Survive",
            "released": 2018,
            "genre": "Comedy, Drama",
            "tagline": "A story of hope and survival",
            "rating": 9.0,
            "box_office": 470000000,
            "language": "Mandarin",
            "country": "China",
            "runtime": 117,
            "plot": "A shopkeeper becomes an illegal drug dealer to help leukemia patients"
        }),
        Node("m35", "Movie", {
            "title": "Better Days",
            "released": 2019,
            "genre": "Crime, Drama, Romance",
            "tagline": "A story of youth and bullying",
            "rating": 7.4,
            "box_office": 230000000,
            "language": "Mandarin",
            "country": "China, Hong Kong",
            "runtime": 135,
            "plot": "A bullied high school student forms an unlikely friendship with a small-time criminal"
        }),
        Node("m36", "Movie", {
            "title": "The Eight Hundred",
            "released": 2020,
            "genre": "Action, Drama, History",
            "tagline": "A heroic last stand",
            "rating": 7.5,
            "box_office": 461000000,
            "language": "Mandarin",
            "country": "China",
            "runtime": 149,
            "plot": "In 1937, Chinese soldiers defend a warehouse during the Battle of Shanghai"
        }),
        Node("m37", "Movie", {
            "title": "Hi, Mom",
            "released": 2021,
            "genre": "Comedy, Drama, Fantasy",
            "tagline": "A time-traveling comedy",
            "rating": 7.8,
            "box_office": 822000000,
            "language": "Mandarin",
            "country": "China",
            "runtime": 128,
            "plot": "A woman travels back in time to meet her late mother"
        }),
        Node("m38", "Movie", {
            "title": "Full River Red",
            "released": 2023,
            "genre": "Comedy, Drama, History",
            "tagline": "A historical mystery",
            "rating": 7.2,
            "box_office": 673000000,
            "language": "Mandarin",
            "country": "China",
            "runtime": 159,
            "plot": "A comedy mystery set during the Song Dynasty"
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
        
        # 中国电影关系
        Relationship("r54", "p22", "m16", "DIRECTED", {}),
        Relationship("r55", "p24", "m16", "ACTED_IN", {"roles": ["Jen Yu"], "billing": 1}),
        Relationship("r56", "p23", "m16", "ACTED_IN", {"roles": ["Li Mu Bai"], "billing": 2}),
        
        Relationship("r57", "p21", "m17", "DIRECTED", {}),
        Relationship("r58", "p23", "m17", "ACTED_IN", {"roles": ["Nameless"], "billing": 1}),
        
        Relationship("r59", "p21", "m18", "DIRECTED", {}),
        Relationship("r60", "p24", "m18", "ACTED_IN", {"roles": ["Mei"], "billing": 1}),
        
        Relationship("r61", "p27", "m19", "DIRECTED", {}),
        Relationship("r62", "p27", "m19", "WROTE", {}),
        
        Relationship("r63", "p26", "m20", "DIRECTED", {}),
        Relationship("r64", "p26", "m20", "ACTED_IN", {"roles": ["Leng Feng"], "billing": 1}),
        Relationship("r65", "p25", "m20", "ACTED_IN", {"roles": ["Big Daddy"], "billing": 2}),
        
        # 更多中国电影关系
        Relationship("r66", "p35", "m22", "DIRECTED", {}),
        Relationship("r67", "p28", "m22", "ACTED_IN", {"roles": ["Juxian"], "billing": 1}),
        Relationship("r68", "p29", "m22", "ACTED_IN", {"roles": ["Cheng Dieyi"], "billing": 2}),
        
        Relationship("r69", "p21", "m23", "DIRECTED", {}),
        Relationship("r70", "p28", "m23", "ACTED_IN", {"roles": ["Songlian"], "billing": 1}),
        
        Relationship("r71", "p21", "m24", "DIRECTED", {}),
        Relationship("r72", "p28", "m24", "ACTED_IN", {"roles": ["Jiazhen"], "billing": 1}),
        
        Relationship("r73", "p21", "m25", "DIRECTED", {}),
        Relationship("r74", "p28", "m25", "ACTED_IN", {"roles": ["Jiu'er"], "billing": 1}),
        
        Relationship("r75", "p22", "m26", "DIRECTED", {}),
        Relationship("r76", "p25", "m26", "ACTED_IN", {"roles": ["Ip Man"], "billing": 1}),
        Relationship("r77", "p30", "m26", "ACTED_IN", {"roles": ["Gong Er"], "billing": 2}),
        
        Relationship("r78", "p25", "m27", "ACTED_IN", {"roles": ["Ip Man"], "billing": 1}),
        
        Relationship("r79", "p25", "m28", "ACTED_IN", {"roles": ["Ip Man"], "billing": 1}),
        
        Relationship("r80", "p37", "m29", "ACTED_IN", {"roles": ["Qin Feng"], "billing": 1}),
        Relationship("r81", "p38", "m29", "ACTED_IN", {"roles": ["Tang Ren"], "billing": 2}),
        
        Relationship("r82", "p37", "m30", "ACTED_IN", {"roles": ["Qin Feng"], "billing": 1}),
        Relationship("r83", "p38", "m30", "ACTED_IN", {"roles": ["Tang Ren"], "billing": 2}),
        
        Relationship("r84", "p33", "m32", "DIRECTED", {}),
        Relationship("r85", "p33", "m32", "ACTED_IN", {"roles": ["Liu Xuan"], "billing": 1}),
        
        Relationship("r86", "p26", "m33", "ACTED_IN", {"roles": ["Yang Rui"], "billing": 1}),
        
        Relationship("r87", "p38", "m34", "ACTED_IN", {"roles": ["Cheng Yong"], "billing": 1}),
        
        Relationship("r88", "p40", "m35", "ACTED_IN", {"roles": ["Chen Nian"], "billing": 1}),
        Relationship("r89", "p41", "m35", "ACTED_IN", {"roles": ["Xiao Bei"], "billing": 2}),
        
        Relationship("r90", "p42", "m36", "ACTED_IN", {"roles": ["Xie Jinyuan"], "billing": 1}),
        
        Relationship("r91", "p39", "m37", "ACTED_IN", {"roles": ["Jia Xiaoling"], "billing": 1}),
        Relationship("r92", "p38", "m37", "ACTED_IN", {"roles": ["Shen Guanglin"], "billing": 2}),
        
        Relationship("r93", "p42", "m38", "DIRECTED", {}),
        Relationship("r94", "p42", "m38", "ACTED_IN", {"roles": ["Zhang Da"], "billing": 1}),
    ]
    
    # 添加所有关系
    for rel in relationships:
        source_node = graph_data.get_node(rel.source_id)
        target_node = graph_data.get_node(rel.target_id)
        if source_node and target_node:
            graph_data.add_relationship(rel)
    
    return graph_data


