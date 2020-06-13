from os import environ
import redis

class Config:
	""" Set flask configuration vars from .env file """

	SECRET_KEY = environ.get('CLV_KEY')
	SESSION_TYPE = 'redis'

	REDIS_HOST= 'localhost'
	REDIS_PORT= 6379
	REDIS_LOCAL_URL = 'redis://{}:{}'.format(REDIS_HOST, REDIS_PORT)
	
	SESSION_REDIS = redis.from_url(environ.get('REDIS_URL', REDIS_LOCAL_URL)) 
